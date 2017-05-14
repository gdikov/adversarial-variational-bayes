from architectures import basic_network
from base_network import BaseNetwork
import keras.backend as K
from edward.models import MultivariateNormalDiag


class Encoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Encoder, self).__init__()
        # TODO: make model parametrisation configurable
        # TODO: make noise dimensionality configurable
        batch_size = inputs.shape[0]
        input_noise = MultivariateNormalDiag(loc=K.zeros(batch_size), scale_diag=K.ones(batch_size))
        self.encoder_input = K.concatenate([inputs, input_noise], axis=1)
        self.latent_factors = basic_network(self.encoder_input, output_shape)

    def get_input(self):
        return self.encoder_input

    def get_output(self):
        return self.latent_factors

    def get_loss(self):
        pass




