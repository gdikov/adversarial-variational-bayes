import logging
import keras.backend as ker

from numpy import pi as pi_const
from keras.layers import Lambda, Concatenate, Multiply, Add, Dense
from keras.models import Input
from keras.models import Model

from architectures import get_network_by_name
from ...utils.config import load_config

config = load_config('global_config.yaml')
logger = logging.getLogger(__name__)


class BaseEncoder(object):
    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic', name='encoder'):
        logger.info("Initialising {} model with {}-dimensional data and {}-dimensional noise input "
                    "and {} dimensional latent output".format(name, data_dim, noise_dim, latent_dim))
        self.name = name
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.network_architecture = network_architecture
        self.data_input = Input(shape=(data_dim,), name='enc_data_input')
        self.standard_normal_sampler = Lambda(self.sample_standard_normal_noise, name='enc_standard_normal_sampler')
        self.standard_normal_sampler2 = Lambda(self.sample_standard_normal_noise, name='enc_standard_normal_sampler2')

    def sample_standard_normal_noise(self, inputs, **kwargs):
        n_samples = kwargs.get('n_samples', ker.shape(inputs)[0])
        n_basis_noise_vectors = kwargs.get('n_basis', -1)
        if n_basis_noise_vectors > 0:
            samples_isotropic = ker.random_normal(shape=(n_samples, n_basis_noise_vectors, self.noise_dim),
                                                  mean=0, stddev=1, seed=config['seed'])
        else:
            samples_isotropic = ker.random_normal(shape=(n_samples, self.noise_dim),
                                                  mean=0, stddev=1, seed=config['seed'])
        op_mode = kwargs.get('mode', 'none')
        if op_mode == 'concatenate':
            concat = Concatenate(axis=1, name='enc_noise_concatenation')([inputs, samples_isotropic])
            return concat
        elif op_mode == 'add':
            resized_noise = Dense(self.data_dim, activation=None, name='enc_resized_noise_sampler')(samples_isotropic)
            added_noise_data = Add(name='enc_adding_noise_data')([inputs, resized_noise])
            return added_noise_data
        return samples_isotropic

    def __call__(self, *args, **kwargs):
        return None


class StandardEncoder(BaseEncoder):
    """
    An Encoder model is trained to parametrise an arbitrary posterior approximate distribution given some 
    input x, i.e. q(z|x). The model takes as input concatenated data samples and arbitrary noise and produces
    a latent encoding:
    
      Data                              Input
     - - - - - - - - -   
       |       Noise                      
       |         |                        
       ----------- <-- concatenation    
            |                           Encoder model
       -----------
       | Encoder |                      
       -----------
            |
        Latent space                    Output
    
    """
    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic'):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the Encoder model
        """
        super(StandardEncoder, self).__init__(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim,
                                              network_architecture=network_architecture, name='Standard Encoder')

        # self.standard_normal_sampler.arguments = {'mode': 'add'}
        noise_input = self.standard_normal_sampler(self.data_input)
        encoder_body_model = get_network_by_name['encoder'][network_architecture](data_dim, noise_dim, latent_dim)
        latent_factors = encoder_body_model([self.data_input, noise_input])
        self.encoder_model = Model(inputs=self.data_input, outputs=latent_factors, name='encoder')

    def __call__(self, *args, **kwargs):
        """
        Make the Encoder model callable on a list of Input layers.
        
        Args:
            *args: a list of input layers from the super-model or numpy arrays in case of test-time inference.
            **kwargs: 

        Returns:
            An Encoder model.
        """
        return self.encoder_model(args[0])


class MomentEstimationEncoder(BaseEncoder):
    """
    An Encoder model is trained to parametrise an arbitrary posterior approximate distribution given some 
    input x, i.e. q(z|x). The model takes as input concatenated data samples and arbitrary noise and produces
    a latent encoding. Additionally the first two moments (mean and variance) are estimated empirically, which is
    necessary for the Adaptive Contrast learning algorithm. Schematically it can be represented as follows:

       Data  Noise
        |      |
        |      |
        |      | 
       -----------
       | Encoder |  ----> empirical mean and variance
       -----------
            |
        Latent space

    """

    def __init__(self, data_dim, noise_dim, noise_basis_dim, latent_dim, network_architecture='mnist'):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            noise_basis_dim: int, noise basis vectors dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the moment estimation Encoder model
        """
        super(MomentEstimationEncoder, self).__init__(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim,
                                                      network_architecture=network_architecture,
                                                      name='Posterior Moment Estimation Encoder')

        models = get_network_by_name['moment_estimation_encoder'][network_architecture](
            data_dim=data_dim, noise_dim=noise_dim, noise_basis_dim=noise_basis_dim, latent_dim=latent_dim)

        data_feature_extraction, noise_basis_extraction = models

        self.standard_normal_sampler.arguments = {'n_basis': noise_basis_dim}
        noise = self.standard_normal_sampler(self.data_input)
        noise_basis_vectors = noise_basis_extraction(noise)

        coefficients_and_z0 = data_feature_extraction(self.data_input)
        coefficients = coefficients_and_z0[:-1]
        z_0 = coefficients_and_z0[-1]

        latent_factors = []
        for i, (a, v) in enumerate(zip(coefficients, noise_basis_vectors)):
            latent_factors.append(Multiply(name='enc_elemwise_coeff_vecs_mult_{}'.format(i))([a, v]))
        latent_factors = Add(name='enc_add_weighted_vecs')(latent_factors)
        latent_factors = Add(name='add_z0_to_linear_combination')([z_0, latent_factors])

        self.standard_normal_sampler2.arguments = {'n_basis': noise_basis_dim, 'n_samples': 100}
        more_noise = self.standard_normal_sampler2(self.data_input)
        sampling_basis_vectors = noise_basis_extraction(more_noise)

        posterior_mean = []
        posterior_var = []
        for i in range(noise_basis_dim):
            # compute empirical mean as the batchsize-wise mean of all sampling vectors for each basis dimension
            mean_basis_vectors_i = Lambda(lambda x: ker.mean(x, axis=0),
                                          name='enc_noise_basis_vectors_mean_{}'.format(i))(sampling_basis_vectors[i])
            # and do the same for the empirical variance and compute similar posterior parametrization for the variance
            var_basis_vectors_i = Lambda(lambda x: ker.var(x, axis=0),
                                         name='enc_noise_basis_vectors_var_{}'.format(i))(sampling_basis_vectors[i])
            # and parametrise the posterior moment as described in the AVB paper
            posterior_mean.append(Lambda(lambda x: x[0] * x[1],
                                         name='enc_moments_mult_mean_{}'.format(i))([coefficients[i],
                                                                                     mean_basis_vectors_i]))

            # compute similar posterior parametrization for the variance
            posterior_var.append(Lambda(lambda x: x[0]*x[0]*x[1],
                                        name='enc_moments_mult_var_{}'.format(i))([coefficients[i],
                                                                                   var_basis_vectors_i]))
        posterior_mean = Add(name='enc_moments_mean')(posterior_mean)
        posterior_mean = Add(name='enc_moments_mean_add_z0')([posterior_mean, z_0])
        posterior_var = Add(name='enc_moments_var')(posterior_var)

        normalised_latent_factors = Lambda(lambda x: (x[0] - x[1]) / ker.sqrt(x[2] + 1e-5),
                                           name='enc_norm_posterior')([latent_factors, posterior_mean, posterior_var])

        log_latent_space = Lambda(lambda x: -0.5 * ker.sum(x**2 + ker.log(2*pi_const), axis=1),
                                  name='enc_log_approx_posterior')(latent_factors)

        log_adaptive_prior = Lambda(lambda x: -0.5 * ker.sum(x[0]**2 + ker.log(x[1]) + ker.log(2*pi_const), axis=1),
                                    name='enc_log_adaptive_prior')([normalised_latent_factors, posterior_var])

        self.encoder_inference_model = Model(inputs=self.data_input, outputs=latent_factors,
                                             name='encoder_inference_model')
        self.encoder_trainable_model = Model(inputs=self.data_input,
                                             outputs=[latent_factors, normalised_latent_factors,
                                                      posterior_mean, posterior_var,
                                                      log_adaptive_prior, log_latent_space],
                                             name='encoder_trainable_model')

    def __call__(self, *args, **kwargs):
        """
        Make the Encoder model callable on a list of Input layers.

        Args:
            *args: a list of input layers from the super-model or numpy arrays in case of test-time inference.
            **kwargs: 

        Returns:
            An Encoder model.
        """
        is_learning = kwargs.get('is_learning', True)
        if is_learning:
            return self.encoder_trainable_model(args[0])
        return self.encoder_inference_model(args[0])


class ReparametrisedGaussianEncoder(BaseEncoder):
    """
    A ReparametrisedGaussianEncoder model is trained to parametrise a Gaussian latent variables:

           Data              
            | 
       -----------
       | Encoder |
       -----------
            |
    mu + sigma * Noise   <--- Reparametrised Gaussian latent space

    """

    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic'):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the reparametrised Gaussian Encoder model
        """
        super(ReparametrisedGaussianEncoder, self).__init__(data_dim=data_dim,
                                                            noise_dim=noise_dim,
                                                            latent_dim=latent_dim,
                                                            network_architecture=network_architecture,
                                                            name='Reparametrised Gaussian Encoder')

        latent_mean, latent_log_var = get_network_by_name['reparametrised_encoder'][network_architecture](
            self.data_input, latent_dim)

        noise = self.standard_normal_sampler(self.data_input)
        latent_factors = Lambda(lambda x: x[0] + ker.exp(x[1] / 2.0) * x[2],
                                name='enc_reparametrised_latent')([latent_mean, latent_log_var, noise])

        self.encoder_inference_model = Model(inputs=self.data_input, outputs=latent_factors, name='encoder_inference')
        self.encoder_learning_model = Model(inputs=self.data_input,
                                            outputs=[latent_factors, latent_mean, latent_log_var],
                                            name='encoder_learning')

    def __call__(self, *args, **kwargs):
        """
        Make the Encoder model callable on a list of Input layers.

        Args:
            *args: a list of input layers from the super-model or numpy arrays in case of test-time inference.

        Keyword Args:
            is_learning: bool, whether the model is used for training or inference. The output is either 
                the latent space or the latent space and the means and variances from which it is reparametrised.  

        Returns:
            An Encoder model.
        """
        is_learninig = kwargs.get('is_learning', True)
        if is_learninig:
            return self.encoder_learning_model(args[0])
        else:
            return self.encoder_inference_model(args[0])
