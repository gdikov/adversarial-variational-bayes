from networks import ReparametrisedGaussianEncoder, Decoder
from losses import VAELossLayer

from keras.models import Input, Model
from keras.optimizers import Adam


class VariationalAutoencoder(object):
    def __init__(self, data_dim, latent_dim):
        encoder = ReparametrisedGaussianEncoder(data_dim=data_dim, noise_dim=latent_dim, latent_dim=latent_dim)
        decoder = Decoder(data_dim=data_dim, latent_dim=latent_dim)

        data_input = Input(shape=(data_dim,), name='vae_data_input')
        noise_input = Input(shape=(latent_dim,), name='vae_noise_input')
        latent_input = Input(shape=(latent_dim,), name='vae_latent_input')

        posterior_approximation, latent_mean, latent_log_var = encoder([data_input, noise_input], is_learning=True)
        reconstruction_log_likelihood = decoder([data_input, posterior_approximation], is_learning=True)

        vae_loss = VAELossLayer(name='vae_loss')([reconstruction_log_likelihood, latent_mean, latent_log_var])

        self._vae_model = Model(inputs=[data_input, noise_input], outputs=vae_loss)
        self._vae_model.compile(optimizer=Adam(lr=1e-3, beta_1=0.5), loss=None)

        # the deployable models
        self._inference_model = Model(inputs=[data_input, noise_input], outputs=posterior_approximation)
        self._generative_model = Model(inputs=latent_input, outputs=decoder(latent_input, is_learning=False))

    def fit(self, data, batch_size=32, epochs=1):
        pass

    def infer(self, data, batch_size=32):
        pass

    def generate(self, n_samples, batch_size=32):
        pass
