import keras.backend as ker
from keras.layers import Layer
from keras.losses import categorical_crossentropy


class AVBLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AVBLossLayer, self).__init__(**kwargs)

    @staticmethod
    def avb_loss(discrim_output_posterior, discrim_output_posterior_prior, data_log_probs):
        # 1/m * sum_{i=1}^m log p(x_i|z), where z = encoder(x_i, epsilon_i)
        reconstruction_log_likelihood = ker.mean(ker.sum(data_log_probs, axis=1))
        # The decoder tries to maximise the reconstruction data log-likelihood
        decoder_loss = -reconstruction_log_likelihood
        # The encoder tries to minimize the discriminator output
        encoder_loss = ker.mean(discrim_output_posterior)
        # The dicriminator loss is the GAN loss with input from the prior and posterior distributions
        discriminator_loss = ker.mean(categorical_crossentropy(y_true=ker.ones_like(discrim_output_posterior),
                                                               y_pred=discrim_output_posterior)
                                      + categorical_crossentropy(y_true=ker.zeros_like(discrim_output_posterior_prior),
                                                                 y_pred=discrim_output_posterior_prior))
        return ker.mean(encoder_loss + decoder_loss + discriminator_loss)

    def call(self, inputs, **kwargs):
        discrim_output_posterior, discrim_output_prior, decoder_output_log_probs = inputs
        loss = self.avb_loss(discrim_output_posterior, discrim_output_prior, decoder_output_log_probs)
        self.add_loss(loss, inputs=inputs)
        # unused output
        return inputs[0]
