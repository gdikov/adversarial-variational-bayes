from architectures import basic_network
from base_network import BaseNetwork
import keras.backend as K
from keras.losses import categorical_crossentropy


class Discraminator(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Discraminator, self).__init__()
        self.pred_posterior = basic_network(K.concatenate((inputs['data'], inputs['posterior']), axis=1), output_shape)
        self.pred_prior = basic_network(K.concatenate((inputs['data'], inputs['prior']), axis=1), output_shape)

    def get_input(self):
        pass

    def get_output(self):
        pass

    def get_loss(self):
        """
        Return the GAN loss for the discriminator network. The objective function to optimise is:
        max_T E_D[E_z~q(z|x)[log(sigma(T(x,z)) + E_z~p(z)[log(1 - sigma(T(x,z)))]] which is maximized if
        T(x,z) produces large positive values for z~q(z|x) and large negative values for z~p(z).
         
        Returns:
            The GAN loss of the discriminator network. 
        """
        return K.mean(categorical_crossentropy(y_true=K.ones_like(self.pred_posterior), y_pred=self.pred_posterior)
                      + categorical_crossentropy(y_true=K.zeros_like(self.pred_prior), y_pred=self.pred_prior))
