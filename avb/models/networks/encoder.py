import logging
from keras.models import Model, Input

from architectures import get_network_by_name

logger = logging.getLogger(__name__)


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
    def __init__(self, data_dim, noise_dim, latent_dim, network_architecture='synthetic'):
        """
        Args:
            data_dim: int, flattened data space dimensionality 
            noise_dim: int, flattened noise space dimensionality
            latent_dim: int, flattened latent space dimensionality
            network_architecture: str, the architecture name for the body of the Encoder model
        """
        logger.info("Initialising Encoder model with {} dimensional data and {} dimensional noise input "
                    "and {} dimensional latent output".format(data_dim, noise_dim, latent_dim))

        data_input = Input(shape=(data_dim,), name='enc_input_data')
        noise_input = Input(shape=(noise_dim,), name='enc_input_noise')

        latent_factors = get_network_by_name['encoder'][network_architecture]([data_input, noise_input], latent_dim)

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
        estimate_moments = kwargs.get('estimate_moments', False)
        if estimate_moments:
            return None
        else:
            return self.encoder_model(args[0])
