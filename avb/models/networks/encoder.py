from architectures import simple_network
from base_network import BaseNetwork
from keras.layers import Concatenate, Dense


class Encoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Encoder, self).__init__()
        self.encoder_input = Concatenate(axis=1, name='data_noise_concat')(inputs)
        latent_factors = Dense(256, activation='relu')(self.encoder_input)
        latent_factors = Dense(256, activation='relu')(latent_factors)
        self.latent_factors = Dense(output_shape, activation=None)(latent_factors)

    def get_output(self):
        return self.latent_factors
