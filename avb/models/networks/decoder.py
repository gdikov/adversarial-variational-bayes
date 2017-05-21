from architectures import simple_network
from base_network import BaseNetwork
from tensorflow.contrib.distributions import Bernoulli
from keras.layers import Lambda, Dense


class Decoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Decoder, self).__init__()
        parametrisation = simple_network(inputs[0], output_shape)
        parametrisation = Dense(output_shape, activation='sigmoid')(parametrisation)
        # self.output_sampler = Lambda(lambda x: Bernoulli(probs=x).sample())(parametrisation)
        self.log_probs = Lambda(lambda x: Bernoulli(probs=parametrisation).log_prob(x))(inputs[1])

    def get_output(self):
        return self.log_probs
