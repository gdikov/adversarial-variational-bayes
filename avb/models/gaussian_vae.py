from __future__ import absolute_import
from builtins import range, next

from keras.models import Model
from keras.optimizers import RMSprop
from tqdm import tqdm

from ..utils.config import load_config
from .losses import VAELossLayer
from .networks import ReparametrisedGaussianEncoder, Decoder
from ..data_iterator import VAEDataIterator
from ..models import BaseVariationalAutoencoder

config = load_config('global_config.yaml')


class GaussianVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, data_dim, latent_dim, resume_from=None,
                 experiment_architecture='synthetic', optimiser_params=None):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            resume_from: str, optional folder name with pre-trained models
            experiment_architecture: str, network architecture descriptor
            optimiser_params: dict, optional optimiser parameters
        """
        self.name = "gaussian_vae"
        self.models_dict = {'vae_model': None}

        self.encoder = ReparametrisedGaussianEncoder(data_dim=data_dim, noise_dim=latent_dim, latent_dim=latent_dim,
                                                     network_architecture=experiment_architecture)
        self.decoder = Decoder(data_dim=data_dim, latent_dim=latent_dim,
                               network_architecture=experiment_architecture)

        # init the base class' inputs and testing models and reuse them
        super(GaussianVariationalAutoencoder, self).__init__(data_dim=data_dim, noise_dim=latent_dim,
                                                             latent_dim=latent_dim, name_prefix=self.name)

        posterior_approximation, latent_mean, latent_log_var = self.encoder(self.data_input, is_learning=True)
        reconstruction_log_likelihood = self.decoder([self.data_input, posterior_approximation], is_learning=True)
        vae_loss = VAELossLayer(name='vae_loss')([reconstruction_log_likelihood, latent_mean, latent_log_var])

        self.vae_model = Model(inputs=self.data_input, outputs=vae_loss)

        if resume_from is not None:
            self.load(resume_from, custom_layers={'VAELossLayer': VAELossLayer})

        optimiser_params = optimiser_params or {'lr': 1e-3}
        self.vae_model.compile(optimizer=RMSprop(**optimiser_params), loss=None)

        self.models_dict['vae_model'] = self.vae_model
        self.data_iterator = VAEDataIterator(data_dim=data_dim, latent_dim=latent_dim, seed=config['seed'])

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the Gaussian Variational Autoencoder onto the training data.
        
        Args:
            data: ndarray, training data
            batch_size: int, number of samples to be fit at one pass
            epochs: int, number of whole-size iterations on the training data
            **kwargs: 

        Returns:
            A training history dict.
        """
        data_iterator, batches_per_epoch = self.data_iterator.iter(data, batch_size, mode='training', shuffle=True)

        history = {'vae_loss': []}
        for _ in tqdm(range(epochs)):
            epoch_loss_history_vae = []
            for it in range(batches_per_epoch):
                data_batch = next(data_iterator)
                loss_autoencoder = self.vae_model.train_on_batch(data_batch[:-1], None)
                epoch_loss_history_vae.append(loss_autoencoder)
            history['vae_loss'].append(epoch_loss_history_vae)

        return history
