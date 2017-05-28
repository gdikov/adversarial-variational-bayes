import logging
from keras.layers import Concatenate, Dense, Input
from keras.models import Model

from architectures import repeat_dense

logger = logging.getLogger(__file__)


class Encoder(object):
    """
    An Encoder model is trained to parametrise an arbitrary posterior approximate distribution given some 
    input x, i.e. q(z|x). The model takes as input concatenated data samples and arbitrary noise and produces
    a latent encoding:
    
      Data     Noise
       |         |
       ----------- <-- concatenation
            | 
       -----------
       | Encoder |
       -----------
            |
        Latent space
    
    """
    def __init__(self, data_dim, noise_dim, latent_dim):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
        """
        logger.info("Initialising Encoder model with {} dimensional data and {} dimensional noise input "
                    "and {} dimensional latent output".format(data_dim, noise_dim, latent_dim))

        data_input = Input(shape=(data_dim,), name='enc_input_data')
        noise_input = Input(shape=(noise_dim,), name='enc_input_noise')
        encoder_input = Concatenate(axis=1, name='enc_data_noise_concat')([data_input, noise_input])

        encoder_body = repeat_dense(encoder_input, num_layers=2, num_units=256, name_prefix='enc_body')

        latent_factors = Dense(latent_dim, activation=None, name='enc_latent')(encoder_body)

        self.encoder_model = Model(inputs=[data_input, noise_input], outputs=latent_factors, name='encoder')

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
