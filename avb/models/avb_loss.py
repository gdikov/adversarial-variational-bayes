import keras.backend as ker
from keras.layers import Layer
from keras.losses import binary_crossentropy


class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)
        # self.trainable = False

    @staticmethod
    def discriminator_loss(discrim_output_prior, discrim_output_posterior):
        # The dicriminator loss is the GAN loss with input from the prior and posterior distributions
        discriminator_loss = ker.mean(binary_crossentropy(y_pred=discrim_output_posterior,
                                                          y_true=ker.ones_like(discrim_output_posterior))
                                      + binary_crossentropy(y_pred=discrim_output_prior,
                                                            y_true=ker.zeros_like(discrim_output_prior)))
        return discriminator_loss

    def call(self, inputs, **kwargs):
        discrim_output_prior, discrim_output_posterior = inputs
        loss = self.discriminator_loss(discrim_output_prior, discrim_output_posterior)
        self.add_loss(loss, inputs=inputs)
        # unused output
        return inputs[0]


class DecoderLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DecoderLossLayer, self).__init__(**kwargs)
        # self.trainable = False

    @staticmethod
    def decoder_loss(data_log_probs, discrim_output_posterior):
        # # The encoder tries to minimize the discriminator output, i.e. to deceive it that this is the prior
        encoder_loss = ker.mean(discrim_output_posterior)
        # 1/m * sum_{i=1}^m log p(x_i|z), where z = encoder(x_i, epsilon_i)
        reconstruction_log_likelihood = ker.mean(ker.sum(data_log_probs, axis=1))
        # The decoder tries to maximise the reconstruction data log-likelihood, hence the minus sign
        decoder_loss = -reconstruction_log_likelihood
        return encoder_loss + decoder_loss

    def call(self, inputs, **kwargs):
        decoder_output_log_probs, discrim_output_posterior = inputs
        loss = self.decoder_loss(decoder_output_log_probs, discrim_output_posterior)
        self.add_loss(loss, inputs=inputs)
        # unused output
        return inputs[0]
