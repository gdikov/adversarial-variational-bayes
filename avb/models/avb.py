import numpy as np

from keras.models import Model, Input
from keras.optimizers import Adam as Optimiser
from freezable_model import FreezableModel

from networks import Encoder, Decoder, Discriminator
from .avb_loss import DiscriminatorLossLayer, DecoderLossLayer
from ..data_iterator import iter_data

SEED = 7
np.random.seed(SEED)


class AdversarialVariationalBayes(object):
    def __init__(self, data_dim, latent_dim=2, noise_dim=None):

        self.data_dim = data_dim
        self.noise_dim = noise_dim or data_dim
        self.latent_dim = latent_dim

        encoder = Encoder(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim)
        decoder = Decoder(latent_dim=latent_dim, data_dim=data_dim)
        discriminator = Discriminator(data_dim=data_dim, latent_dim=latent_dim)

        data_input = Input(shape=(data_dim,), name='avb_data_input')
        noise_input = Input(shape=(noise_dim,), name='avb_noise_input')
        prior_input = Input(shape=(latent_dim,), name='avb_latent_prior_input')

        posterior_approximation = encoder([data_input, noise_input])
        reconstruction_log_likelihood = decoder([data_input, posterior_approximation], is_learning=True)
        discriminator_output_prior = discriminator([data_input, prior_input])
        discriminator_output_posterior = discriminator([data_input, posterior_approximation])

        discriminator_loss = DiscriminatorLossLayer(name='disc_loss')([discriminator_output_prior,
                                                                       discriminator_output_posterior])
        decoder_loss = DecoderLossLayer(name='dec_loss')([reconstruction_log_likelihood,
                                                          discriminator_output_posterior])

        self._avb_disc_train = FreezableModel(inputs=[data_input, noise_input, prior_input],
                                              outputs=discriminator_loss, name_prefix=['disc'])
        self._avb_dec_train = FreezableModel(inputs=[data_input, noise_input, prior_input],
                                             outputs=decoder_loss, name_prefix=['dec', 'enc'])

        self._avb_disc_train.freeze()
        self._avb_dec_train.unfreeze()
        self._avb_dec_train.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None, metrics=['accuracy'])

        self._avb_disc_train.unfreeze()
        self._avb_dec_train.freeze()
        self._avb_disc_train.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None)

        self._inference_model = Model(inputs=[data_input, noise_input], outputs=posterior_approximation)
        self._generative_model = Model(inputs=prior_input, outputs=decoder(prior_input, is_learning=False))
        # from keras.utils import plot_model
        # plot_model(self._avb)

    def fit(self, data, batch_size=32, epochs=1):
        data_iterator_decoder = iter_data(data, batch_size, mode='training', seed=SEED,
                                          latent_dim=self.latent_dim, input_noise_dim=self.noise_dim)
        data_iterator_discriminator = iter_data(data, batch_size, mode='training', seed=SEED,
                                                latent_dim=self.latent_dim, input_noise_dim=self.noise_dim)
        for i in range(epochs):
            self._avb_dec_train.fit_generator(data_iterator_decoder,
                                              steps_per_epoch=1, epochs=1, workers=1, verbose=2)
            self._avb_disc_train.fit_generator(data_iterator_discriminator,
                                               steps_per_epoch=1, epochs=1, workers=1, verbose=2)

    def infer(self, data, batch_size=32):
        data_iterator = iter_data(data, batch_size, mode='inference', seed=SEED, input_noise_dim=self.noise_dim)
        latent_samples = self._inference_model.predict_generator(data_iterator, steps=(data.shape[0] // batch_size))
        return latent_samples

    def generate(self, n_samples, batch_size=32):
        n_samples_per_axis = complex(int(np.sqrt(n_samples)))
        data = np.mgrid[-3:3:n_samples_per_axis, -3:3:n_samples_per_axis].reshape(2, -1).T
        data_iterator = iter_data(data, batch_size=batch_size, mode='generation', seed=SEED,
                                  latent_dim=self.latent_dim)
        data_probs = self._generative_model.predict_generator(data_iterator, steps=(data.shape[0] // batch_size))
        print(data_probs)
        sampled_data = np.random.binomial(1, p=data_probs, size=data_probs.shape)
        return sampled_data
