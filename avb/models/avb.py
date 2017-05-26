import numpy as np
np.random.seed(7)
from keras.models import Model, Input
from keras.optimizers import Adam as Optimiser

from networks import Encoder, Decoder, Discriminator
from .avb_loss import AVBLossLayer


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

        avb_loss = AVBLossLayer()([discriminator_output_prior,
                                   discriminator_output_posterior,
                                   reconstruction_log_likelihood])

        self._avb = Model(inputs=[data_input, noise_input, prior_input], outputs=avb_loss)
        self._avb.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None)

        self._inference_model = Model(inputs=[data_input, noise_input], outputs=posterior_approximation)
        self._generative_model = Model(inputs=prior_input, outputs=decoder(prior_input, is_learning=False))
        # from keras.utils import plot_model
        # plot_model(self._avb)

    def _iter_data(self, data=None, batch_size=32, mode='training'):
        if mode == 'inference':
            data_size = data.shape[0]
            batches = np.split(np.arange(data_size), data_size // batch_size)
            while True:
                for batch_indices in batches:
                    random_noise_data = np.random.standard_normal(size=(batch_size, self.noise_dim))
                    yield [data[batch_indices], random_noise_data]
        elif mode == 'generation':
            while True:
                random_noise_prior = np.random.uniform([-0.6, -10], [-1, -7], size=(batch_size, self.latent_dim))
                yield random_noise_prior
        else:
            data_size = data.shape[0]
            while True:
                indices = np.random.choice(data_size, size=batch_size, replace=True)
                random_noise_data = np.random.standard_normal(size=(batch_size, self.noise_dim))
                random_noise_prior = np.random.standard_normal(size=(batch_size, self.latent_dim))
                yield [data[indices], random_noise_data, random_noise_prior], None

    def fit(self, data, batch_size=32, epochs=1):
        self._avb.fit_generator(self._iter_data(data, batch_size), steps_per_epoch=8, epochs=epochs, workers=1)

    def infer(self, data, batch_size=32):
        data_size = data.shape[0]
        latent_vars = self._inference_model.predict_generator(self._iter_data(data, batch_size, mode='inference'),
                                                              steps=data_size//batch_size)
        return latent_vars

    def generate(self, n_points=10):
        data_probs = self._generative_model.predict_generator(self._iter_data(batch_size=n_points,
                                                                              mode='generation'), steps=1)
        sampled_data = np.random.binomial(1, p=data_probs, size=(n_points, self.data_dim))
        return sampled_data
