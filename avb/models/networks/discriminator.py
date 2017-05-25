from architectures import simple_network
from base_network import BaseNetwork
from keras.layers import Concatenate, Dense
from keras.models import Model, Input


class Discriminator(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Discriminator, self).__init__()

        discriminator_input = Input(shape=(4+2,))
        network_body = Dense(256, activation='relu')(discriminator_input)
        network_body = Dense(256, activation='relu')(network_body)
        regression_output = Dense(output_shape, activation='sigmoid', name='discriminator_output')(network_body)

        discriminator_model = Model(inputs=discriminator_input, outputs=regression_output)

        data_input, posterior_input, prior_input = inputs

        concat_posterior = Concatenate(axis=1)([data_input, posterior_input])
        concat_prior = Concatenate(axis=1)([data_input, prior_input])

        self.d_model_post = discriminator_model(concat_posterior)
        self.d_model_prior = discriminator_model(concat_prior)


    def get_output(self):
        return self.d_model_post, self.d_model_prior

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
