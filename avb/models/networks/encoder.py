from architectures import simple_network
from base_network import BaseNetwork
from keras.layers import Concatenate


class Encoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Encoder, self).__init__()
        self.encoder_input = Concatenate(axis=1, name='data_noise_concat')(inputs)
        self.latent_factors = simple_network(self.encoder_input, output_shape)

    def get_output(self):
        return self.latent_factors
