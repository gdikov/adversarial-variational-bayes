from tensorflow.contrib.distributions import Bernoulli
from keras.layers import Lambda, Dense
from keras.models import Model, Input

from architectures import get_network_by_name


class Decoder(object):
    """
    A Decoder model has inputs comprising of a latent encoding given by an Encoder model, a prior sampler 
    or other custom input and the raw Encoder data input, which is needed to estimate the reconstructed 
    data log likelihood. It can be visualised as:
     
      Data    Latent
       |        |
       |    -----------
       |    | Decoder |
       |    -----------
       |        |
       |      Output
       |    probability    --->  Generated data
       |        |
       ---> Log Likelihood ---> -(reconstruction loss)
    
    Note that the reconstruction loss is not used when the model training ends. It serves only the purpose to 
    define a measure of loss which is optimised. 
    """
    def __init__(self, latent_dim, data_dim, network_architecture='synthetic'):
        """
        Args:
            latent_dim: int, the flattened dimensionality of the latent space 
            data_dim: int, the flattened dimensionality of the output space (data space)
            network_architecture: str, the architecture name for the body of the Decoder model
        """
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        # NOTE: all decoder layers have names prefixed by `dec`.
        # This is essential for the partial model freezing during training.

        real_data = Input(shape=(self.data_dim,), name='dec_ll_estimator_data_input')
        latent_encoding = Input(shape=(self.latent_dim,), name='dec_latent_input')

        generator_body = get_network_by_name['decoder'][network_architecture](latent_encoding)

        sampler_params = Dense(self.data_dim, activation='sigmoid', name='dec_sampler_params')(generator_body)

        # a probability clipping is necessary for the Bernoulli `log_prob` property produces NaNs in the border cases.
        sampler_params = Lambda(lambda x: 1e-6 + (1 - 2e-6) * x, name='dec_probs_clipper')(sampler_params)

        log_probs = Lambda(lambda x: Bernoulli(probs=x[0], name='dec_bernoulli').log_prob(x[1]),
                           name='dec_bernoulli_logprob')([sampler_params, real_data])

        self.generator = Model(inputs=latent_encoding, outputs=sampler_params, name='dec_sampling')
        self.ll_estimator = Model(inputs=[real_data, latent_encoding], outputs=log_probs, name='dec_trainable')

    def __call__(self, *args, **kwargs):
        """
        Make the Decoder model callable on lists of Input layers or tensors.
        
        Args:
            *args: a list of input layers or tensors or numpy arrays, or a single input layer, tensor or numpy array.
        Keyword Args:
            is_learning: bool, whether the model is used for training or data generation. The output is either 
                the reconstruction log likelihood or the output probabilities in the data space respectively.

        Returns:
            A Decoder model in `training` or `data generation` mode. 
        """
        is_learninig = kwargs.get('is_learning', True)
        if is_learninig:
            return self.ll_estimator(args[0])
        else:
            return self.generator(args[0])
