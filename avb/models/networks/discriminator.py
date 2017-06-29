from __future__ import absolute_import

import logging

import keras.backend as ker
from keras.layers import Lambda
from keras.models import Model, Input

from .architectures import get_network_by_name
from ...utils.config import load_config

config = load_config('global_config.yaml')
logger = logging.getLogger(__name__)


class BaseDiscriminator(object):
    def __init__(self, data_dim, latent_dim, network_architecture='synthetic', name='discriminator'):
        logger.info("Initialising {} model with {}-dimensional data "
                    "and {}-dimensional prior/latent input.".format(name, data_dim, latent_dim))
        self.name = name
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.network_architecture = network_architecture
        self.data_input = Input(shape=(data_dim,), name='disc_data_input')
        self.latent_input = Input(shape=(latent_dim,), name='disc_latent_input')
        self.prior_sampler = Lambda(self.sample_adaptive_normal_noise, name='disc_prior_sampler')
        self.discriminator_from_prior_model = None
        self.discriminator_from_posterior_model = None

    def sample_adaptive_normal_noise(self, inputs, **kwargs):
        if isinstance(inputs, list):
            mu, sigma2 = inputs
            n_samples = kwargs.get('n_samples', ker.shape(mu)[0])
            samples_isotropic = ker.random_normal(shape=(n_samples, self.latent_dim),
                                                  mean=0, stddev=1, seed=config['seed'])
            samples = mu + ker.sqrt(sigma2) * samples_isotropic
            return samples
        else:
            samples_isotropic = ker.random_normal(shape=(ker.shape(inputs)[0], self.latent_dim),
                                                  mean=0, stddev=1, seed=config['seed'])
            return samples_isotropic

    def __call__(self, *args, **kwargs):
        return None


class Discriminator(BaseDiscriminator):
    """
    Discriminator model is adversarially trained against the encoder in order to account 
    for a D_KL(q(z|x) || p(z)) term in the variational loss (see AVB paper, page 3). The discriminator
    architecture takes as input samples from the joint probability distribution of the data `x` and a approximate
    posterior `z` and from the joint of the data and the prior over `z`:
     
             -----------
       ----> | Encoder |
       |     -----------
       |         |
       |    Approx. posterior --> | 
       |                          |---> (x, z') --|
       -------------------------> |               |
       |                                          |     -----------------
      Data                                        | --> | Discriminator | --> T(x,z) regression output
       |                                          |     -----------------
       -------------------------> |               |
                                  |---> (x, z)  --|
       Prior p(z): N(0,I) ------> |
       
    """
    def __init__(self, data_dim, latent_dim, network_architecture='synthetic'):
        """
        Args:
            data_dim: int, the flattened dimensionality of the data space
            latent_dim: int, the flattened dimensionality of the latent space
            network_architecture: str, the architecture name for the body of the Discriminator model
        """
        super(Discriminator, self).__init__(data_dim=data_dim, latent_dim=latent_dim,
                                            network_architecture=network_architecture,
                                            name='Standard Discriminator')

        discriminator_model = get_network_by_name['discriminator'][network_architecture](self.data_dim, self.latent_dim)
        prior_distribution = self.prior_sampler(self.data_input)
        from_prior_output = discriminator_model([self.data_input, prior_distribution])
        self.discriminator_from_prior_model = Model(inputs=self.data_input, outputs=from_prior_output,
                                                    name='discriminator_from_prior')
        from_posterior_output = discriminator_model([self.data_input, self.latent_input])
        self.discriminator_from_posterior_model = Model(inputs=[self.data_input, self.latent_input],
                                                        outputs=from_posterior_output,
                                                        name='discriminator_from_posterior')

    def __call__(self, *args, **kwargs):
        """
        Make the Discriminator model callable on a list of Inputs (coming from the AVB model)
        
        Args:
            *args: a list of Input layers
            **kwargs: 

        Returns:
            A trainable Discriminator model. 
        """
        from_posterior = kwargs.get('from_posterior', False)
        if not from_posterior:
            return self.discriminator_from_prior_model(args[0])
        return self.discriminator_from_posterior_model(args[0])


class AdaptivePriorDiscriminator(BaseDiscriminator):
    """
    Discriminator model is adversarially trained against the encoder in order to account 
    for a D_KL(q(z|x) || p(z)) term in the variational loss (see AVB paper, page 3). The discriminator
    architecture takes as input samples from the joint probability distribution of the data `x` and a approximate
    posterior `z` and from the joint of the data and the prior over `z`:
    
    <------------------------|
    |        ----------- m,s |
    |  ----> | Encoder | ---->      Encoder with mean and variance moment estimation
    |  |     -----------
    |  |         |
    |  |    Approx. posterior --> | 
    |  |                          |---> (x, z') --|
    |  -------------------------> |               |
    |  |                                          |     -----------------
    | Data  <---- Input                           | --> | Discriminator | --> T(x,z) regression output
    |  |                                          |     -----------------
    |  -------------------------> |               |
    |                             |---> (x, z)  --|
    ---> Prior p(z): N(m,sI) ---> |

    """

    def __init__(self, data_dim, latent_dim, network_architecture='synthetic'):
        """
        Args:
            data_dim: int, the flattened dimensionality of the data space
            latent_dim: int, the flattened dimensionality of the latent space
            network_architecture: str, the architecture name for the body of the Discriminator model
        """
        super(AdaptivePriorDiscriminator, self).__init__(data_dim=data_dim, latent_dim=latent_dim,
                                                         network_architecture=network_architecture,
                                                         name='Adaptive Prior Discriminator')

        self.prior_mean = Input(shape=(latent_dim,), name='disc_prior_mean_input')
        self.prior_var = Input(shape=(latent_dim,), name='disc_prior_var_input')
        discriminator_model = get_network_by_name['adaptive_prior_discriminator'][network_architecture](data_dim,
                                                                                                        latent_dim)
        # self.prior_sampler.arguments = {'mean': self.prior_mean, 'variance': self.prior_var}
        prior_distribution = self.prior_sampler([self.prior_mean, self.prior_var])
        from_prior_output = discriminator_model([self.data_input, prior_distribution])
        self.discriminator_from_prior_model = Model(inputs=[self.data_input, self.prior_mean, self.prior_var],
                                                    outputs=from_prior_output,
                                                    name='discriminator_from_prior')
        from_posterior_output = discriminator_model([self.data_input, self.latent_input])
        self.discriminator_from_posterior_model = Model(inputs=[self.data_input, self.latent_input],
                                                        outputs=from_posterior_output,
                                                        name='discriminator_from_posterior')

    def __call__(self, *args, **kwargs):
        """
        Make the Discriminator model callable on a list of Inputs (coming from the AVB model)

        Args:
            *args: a list of Input layers
            **kwargs: 

        Returns:
            A trainable Discriminator model. 
        """
        from_posterior = kwargs.get('from_posterior', False)
        if not from_posterior:
            return self.discriminator_from_prior_model(args[0])
        return self.discriminator_from_posterior_model(args[0])
