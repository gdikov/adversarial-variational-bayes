import numpy as np

from keras.models import Model
from keras.layers import Input, Concatenate
from keras.optimizers import Adam as Optimiser

from networks import Encoder, Decoder, Discriminator
from .avb_loss import AVBLossLayer


class AdversarialVariationalBayes(object):
    def __init__(self, data_dim, latent_dim=2, noise_dim=None):

        self.data_dim = data_dim
        self.noise_dim = noise_dim or data_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(data_dim=data_dim, noise_dim=noise_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=data_dim)
        self.discriminator = Discriminator(data_dim=data_dim, latent_dim=latent_dim)

        data_input = Input(shape=(data_dim,), name='avb_data_input')
        noise_input = Input(shape=(noise_dim,), name='avb_noise_input')
        prior_input = Input(shape=(latent_dim,), name='avb_latent_prior_input')

        posterior_approximation = self.encoder([data_input, noise_input])
        reconstruction_log_likelihood = self.decoder([data_input, posterior_approximation], is_learning=True)
        discriminator_output_posterior = self.discriminator(Concatenate(axis=1)([data_input, posterior_approximation]))
        discriminator_output_prior = self.discriminator(Concatenate(axis=1)([data_input, prior_input]))

        avb_loss = AVBLossLayer()([discriminator_output_posterior,
                                   discriminator_output_prior,
                                   reconstruction_log_likelihood])

        self.avb = Model(inputs=[data_input, noise_input, prior_input], outputs=avb_loss)
        self.avb.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5, epsilon=1e-3), loss=None)

    def _iter_data(self, data=None, batch_size=32, mode='training'):
        if mode == 'inference':
            data_size = data.shape[0]
            while True:
                indices = np.random.choice(data_size, size=batch_size, replace=True)
                random_noise_data = np.random.standard_normal(size=(batch_size, self.noise_dim))
                yield [data[indices], random_noise_data], None
        elif mode == 'generation':
            random_noise_prior = np.random.standard_normal(size=(batch_size, self.latent_dim))
            yield [random_noise_prior], None
        else:
            data_size = data.shape[0]
            while True:
                indices = np.random.choice(data_size, size=batch_size, replace=True)
                random_noise_data = np.random.standard_normal(size=(batch_size, self.noise_dim))
                random_noise_prior = np.random.standard_normal(size=(batch_size, self.latent_dim))
                yield [data[indices], random_noise_data, random_noise_prior], None

    def fit(self, data, batch_size=32, epochs=1):
        self.avb.fit_generator(self._iter_data(data, batch_size), steps_per_epoch=32, epochs=epochs, workers=1)

    # def infer(self, data, batch_size=32):
    #     latent_vars = self.encoder.predict_generator(self._iter_data(data, batch_size, mode='inference'), steps=1)
    #     return latent_vars
    #
    # def generate(self, n_points=10):
    #     data = generative_sampler.predict_generator(self._iter_data(batch_size=n_points, mode='generation'), steps=1)
    #     return data
