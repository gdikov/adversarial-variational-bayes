from architectures import basic_network
from base_network import BaseNetwork


class Encoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Encoder, self).__init__()
        self.latent_factors = basic_network(inputs, output_shape)

    def get_input(self):
        return self.encoder_input

    def get_output(self):
        return self.latent_factors

    def get_loss(self):
        pass




