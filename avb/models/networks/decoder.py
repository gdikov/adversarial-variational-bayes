from architectures import simple_network
from base_network import BaseNetwork
from tensorflow.contrib.distributions import Bernoulli
from keras.layers import Lambda, Dense


class Decoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Decoder, self).__init__()
        parametrisation = Dense(256, activation='relu')(inputs[0])
        parametrisation = Dense(256, activation='relu')(parametrisation)
        parametrisation = 1e-6 + (1 - 2*1e-6) * Dense(output_shape, activation='sigmoid')(parametrisation)
        # self.output_sampler = Lambda(lambda x: Bernoulli(probs=x).sample())(parametrisation)
        self.log_probs = Lambda(lambda x: Bernoulli(probs=parametrisation, validate_args=False).log_prob(x))(inputs[1])

    def get_output(self):
        return self.log_probs
