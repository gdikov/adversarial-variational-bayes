from architectures import simple_network
from base_network import BaseNetwork
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Concatenate, Dense


class Discriminator(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Discriminator, self).__init__()
        self.input_posterior = Concatenate(axis=1)([inputs[0], inputs[1]])
        self.input_prior = Concatenate(axis=1)([inputs[0], inputs[2]])
        self.pred_posterior = simple_network(self.input_posterior, output_shape)
        self.pred_prior = simple_network(self.input_prior, output_shape)

    def get_output(self):
        return Dense(1, activation='sigmoid', name='output_discr_prior')(self.pred_prior), \
               Dense(1, activation='sigmoid', name='output_discr_posterior')(self.pred_posterior)

    # def get_loss(self, y_true, y_pred):
    #     """
    #     Return the GAN loss for the discriminator network. The objective function to optimise is:
    #     max_T E_D[E_z~q(z|x)[log(sigma(T(x,z)) + E_z~p(z)[log(1 - sigma(T(x,z)))]] which is maximized if
    #     T(x,z) produces large positive values for z~q(z|x) and large negative values for z~p(z).
    #
    #     Returns:
    #         The GAN loss of the discriminator network.
    #     """
    #     return K.mean(categorical_crossentropy(y_true=K.ones_like(self.pred_posterior), y_pred=self.pred_posterior)
    #                   + categorical_crossentropy(y_true=K.zeros_like(self.pred_prior), y_pred=self.pred_prior))
