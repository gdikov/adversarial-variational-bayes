from __future__ import absolute_import

import keras.backend as ker

from keras.layers import Layer
from keras.losses import binary_crossentropy
from numpy import prod


class AVBDiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AVBDiscriminatorLossLayer, self).__init__(**kwargs)
        # self.trainable = False

    @staticmethod
    def discriminator_loss(discrim_output_prior, discrim_output_posterior, from_logits=False):
        if from_logits:
            discrim_output_posterior = ker.sigmoid(discrim_output_posterior)
            discrim_output_prior = ker.sigmoid(discrim_output_prior)
        # The dicriminator loss is the GAN loss with input from the prior and posterior distributions
        discriminator_loss = ker.mean(binary_crossentropy(y_pred=discrim_output_posterior,
                                                          y_true=ker.ones_like(discrim_output_posterior))
                                      + binary_crossentropy(y_pred=discrim_output_prior,
                                                            y_true=ker.zeros_like(discrim_output_prior)))
        return discriminator_loss

    def call(self, inputs, **kwargs):
        discrim_output_prior, discrim_output_posterior = inputs
        is_in_logits = kwargs.get('is_in_logits', True)
        loss = self.discriminator_loss(discrim_output_prior, discrim_output_posterior, from_logits=is_in_logits)
        self.add_loss(loss, inputs=inputs)
        # unused output
        return loss


class AVBEncoderDecoderLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AVBEncoderDecoderLossLayer, self).__init__(**kwargs)
        self.use_adaptive_contrast = kwargs.get('use_adaptive_constrast', False)
        # self.trainable = False

    @staticmethod
    def decoder_loss(data_log_probs, discrim_output_posterior, log_adaptive_prior=None, log_latent_space=None):
        norm_factor = 1. / prod(ker.int_shape(data_log_probs)[1:])
        # # The encoder tries to minimize the discriminator output, i.e. to deceive it that this is the prior
        if log_adaptive_prior is None or log_latent_space is None:
            kl_divergence = ker.mean(discrim_output_posterior)
        else:
            kl_divergence = ker.mean(discrim_output_posterior) + \
                            log_adaptive_prior - log_latent_space
        # 1/m * sum_{i=1}^m log p(x_i|z), where z = encoder(x_i, epsilon_i)
        reconstruction_error = ker.mean(-ker.sum(data_log_probs, axis=1))
        # elbo = -kl_divergence - reconstruction_error
        # minimise the divergence and reconstruction error (equivalent ot maximise reconstruction likelihood)
        return norm_factor * ker.mean(kl_divergence + reconstruction_error)

    def call(self, inputs, **kwargs):
        use_adaptive_contrast = kwargs.get('use_adaptive_contrast', False)
        if use_adaptive_contrast:
            decoder_output_log_probs, discrim_output_posterior, \
                log_adaptive_prior, log_latent_space = inputs
            loss = self.decoder_loss(decoder_output_log_probs, discrim_output_posterior,
                                     log_adaptive_prior, log_latent_space)
        else:
            decoder_output_log_probs, discrim_output_posterior = inputs
            loss = self.decoder_loss(decoder_output_log_probs, discrim_output_posterior)

        self.add_loss(loss, inputs=inputs)
        # unused output
        return loss


class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    @staticmethod
    def vae_loss(data_log_probs, mean, log_var):
        # 1/m * sum_{i=1}^m log p(x_i|z), where z = encoder(x_i, epsilon_i)
        reconstruction_log_likelihood = ker.mean(ker.sum(data_log_probs, axis=1))
        # The decoder tries to maximise the reconstruction data log-likelihood, hence the minus sign
        decoder_loss = -reconstruction_log_likelihood
        # The encoder loss, i.e. the analytically solvable KL term in the ELBO for multivariate Normal distributions
        # is D_KL(N_0 || N_1) = 0.5 * [tr(Sigma_1^-1 * Sigma_0) + (mu_1 - mu_0)^T Sigma_1^-1 (mu_1 - mu_0) - k
        #                              + ln(det(Sigma_1) / det(Sigma_0))]
        # where k is the dimensionality of the MVNs, and in the VAE case Sigma_1 = I_k and mu_1 = 0
        encoder_loss = -0.5 * ker.sum(1 - ker.square(mean) - ker.exp(log_var) + log_var, axis=-1)
        return ker.mean(encoder_loss + decoder_loss)

    def call(self, inputs, **kwargs):
        decoder_output_log_probs, mean, log_var = inputs
        loss = self.vae_loss(decoder_output_log_probs, mean, log_var)
        self.add_loss(loss, inputs=inputs)
        # unused output
        return loss
