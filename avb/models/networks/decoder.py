from architectures import basic_network
from base_network import BaseNetwork
import keras.backend as K
from edward.models import Bernoulli


class Decoder(object, BaseNetwork):
    def __init__(self, inputs, output_shape):
        super(Decoder, self).__init__()
        # TODO: make model parametrisation configurable
        self.parametrisation = basic_network(inputs, output_shape)
        self.log_probs = Bernoulli(logits=self.parametrisation)

    def get_input(self):
        pass

    def get_loss(self):
        """
        Return the reconstruction loss: -E_{z~q(z|x)}[log(p(x|z))]. Note the minus sign before the likelihood term.
        It is needed since the objective is to maximise the likelihood of the generated data under the inferred 
        latent variables.

        Returns:
            Loss object of the decoder network.
        Notes:
            In the AVB context the regularisation term D_{KL}(q(z|x)||p(z)) is approximated by the 
            discriminator network and hence is omitted in this loss definition. Gaussian inference models, 
            such as classical VAEs are able to compute this term analytically and hence can be added
            directly into this loss. 
        """
        # Since we compute p(x|z) in log space and we assume independence between the covariates of each sample x_i,
        # we sum the logs of each predicted component and take the mean of all samples in a batch.
        reconstruction_log_likelihood = self.log_probs
        return K.mean(K.sum(reconstruction_log_likelihood, axis=1))





