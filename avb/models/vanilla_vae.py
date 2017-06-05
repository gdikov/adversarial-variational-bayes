from tqdm import tqdm
from networks import ReparametrisedGaussianEncoder, Decoder
from losses import VAELossLayer
from ..models import BaseVariationalAutoencoder

from keras.models import Model
from keras.optimizers import Adam
from ..data_iterator import VAEDataIterator
from utils.config import load_config

config = load_config('global_config.yaml')


class GaussianVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, data_dim, latent_dim, resume_from=None, deployable_models_only=False):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            resume_from: str, optional folder name with pre-trained models 
            deployable_models_only: bool, whether only the inference and generative models should be instantiated
        """
        self.encoder = ReparametrisedGaussianEncoder(data_dim=data_dim, noise_dim=latent_dim, latent_dim=latent_dim)
        self.decoder = Decoder(data_dim=data_dim, latent_dim=latent_dim)
        self.models_dict = {'deployable': {'inference_model': None, 'generative_model': None},
                            'trainable': {'vae_model': None}}
        # init the base class' inputs and deployable models and reuse them in the paer
        super(GaussianVariationalAutoencoder, self).__init__(data_dim=data_dim, noise_dim=latent_dim,
                                                             latent_dim=latent_dim, name_prefix='vae',
                                                             resume_from=resume_from,
                                                             deployable_models_only=deployable_models_only)
        if resume_from is None:
            posterior_approximation, latent_mean, latent_log_var = self.encoder([self.data_input, self.noise_input],
                                                                                is_learning=True)
            reconstruction_log_likelihood = self.decoder([self.data_input, posterior_approximation], is_learning=True)
            vae_loss = VAELossLayer(name='vae_loss')([reconstruction_log_likelihood, latent_mean, latent_log_var])
            self.vae_model = Model(inputs=[self.data_input, self.noise_input], outputs=vae_loss)
            self.vae_model.compile(optimizer=Adam(lr=1e-3), loss=None)

        self.models_dict['trainable']['vae_model'] = self.vae_model
        self.data_iterator = VAEDataIterator(data_dim=data_dim, latent_dim=latent_dim, noise_dim=latent_dim,
                                             seed=config['general']['seed'], noise_distribution='normal')

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the Gaussian Variational Autoencoder onto the training data.
        
        Args:
            data: ndarray, training data
            batch_size: int, number of samples to be fit at one pass
            epochs: int, number of whole-size iterations on the training data
            **kwargs: 

        Returns:

        """
        data_iterator, batches_per_epoch = self.data_iterator.iter(data, batch_size, mode='training', shuffle=True)

        history = {'vae_loss': []}
        for _ in tqdm(xrange(epochs)):
            epoch_loss_history_vae = []
            for it in xrange(batches_per_epoch):
                data_batch = data_iterator.next()
                loss_autoencoder = self.vae_model.train_on_batch(data_batch[:-1], None)
                epoch_loss_history_vae.append(loss_autoencoder)
            history['vae_loss'].append(epoch_loss_history_vae)

        return history
