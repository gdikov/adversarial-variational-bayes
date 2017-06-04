import logging
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.backend import exp

from architectures import synthetic_encoder

logger = logging.getLogger(__name__)


class ReparametrisedGaussianEncoder(object):
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

    def __init__(self, data_dim, noise_dim, latent_dim):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
        """
        logger.info("Initialising Reparametrised Gaussian Encoder model with {} dimensional data "
                    "and {} dimensional latent output".format(data_dim, noise_dim, latent_dim))

        data_input = Input(shape=(data_dim,), name='rep_enc_input_data')
        noise_input = Input(shape=(noise_dim,), name='rep_enc_input_noise')

        encoder_body = synthetic_encoder([data_input, noise_input], latent_dim)
        latent_mean = Dense(latent_dim, activation=None, name='rep_enc_mean')(encoder_body)
        # since the variance must be positive and this is not easy to restrict, interpret it in the log domain
        latent_log_var = Dense(latent_dim, activation=None, name='rep_enc_var')(encoder_body)

        latent_factors = Lambda(lambda x: x[0] + exp(x[1] / 2.0) * x[2],
                                name='rep_enc_reparametrised_latent')([latent_mean, latent_log_var, noise_input])

        self.encoder_inference_model = Model(inputs=[data_input, noise_input], outputs=latent_factors,
                                   name='reparametrised_encoder')
        self.encoder_learning_model = Model(inputs=[data_input, noise_input],
                                            outputs=[latent_factors, latent_mean, latent_log_var])

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
