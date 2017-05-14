from keras.layers import Dense
from base_network import BaseNetwork
import keras.backend as K
from edward.models import MultivariateNormalDiag


class Encoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Encoder, self).__init__()
        # TODO: make model parametrisation configurable
        self.parametrisation = basic_encoder(inputs, output_shape)
        # TODO: make noise dimensionality configurable
        batch_size = inputs.shape[0]
        input_noise = MultivariateNormalDiag(loc=K.zeros(batch_size), scale_diag=K.ones(batch_size))
        encoder_input = K.concatenate([inputs, input_noise], axis=1)
        self.latent_factors = basic_encoder(encoder_input, output_shape)

    def get_input(self):
        pass

    def get_output(self):
        pass

    def get_loss(self):
        pass


def basic_encoder(inputs, output_shape, n_hidden=2):
    h = Dense(512, activation='relu')(inputs)
    for i in xrange(n_hidden - 1):
        h = Dense(512, activation='relu')(h)
    h = Dense(output_shape)(h)
    return h

def resnet_encoder(inputs, output_shape, )