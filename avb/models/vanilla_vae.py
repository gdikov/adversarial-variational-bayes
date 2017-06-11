from keras.models import Model
from keras.optimizers import RMSprop
from tqdm import tqdm

from ..utils.config import load_config
from losses import VAELossLayer
from networks import ReparametrisedGaussianEncoder, Decoder
from ..data_iterator import VAEDataIterator
from ..models import BaseVariationalAutoencoder

config = load_config('global_config.yaml')


class GaussianVariationalAutoencoder(BaseVariationalAutoencoder):
    def __init__(self, data_dim, latent_dim, resume_from=None, deployable_models_only=False,
                 experiment_architecture='synthetic'):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            resume_from: str, optional folder name with pre-trained models 
            deployable_models_only: bool, whether only the inference and generative models should be instantiated
            experiment_architecture: str, network architecture descriptor
        """
        self.encoder = ReparametrisedGaussianEncoder(data_dim=data_dim, noise_dim=latent_dim, latent_dim=latent_dim,
                                                     network_architecture=experiment_architecture)
        self.decoder = Decoder(data_dim=data_dim, latent_dim=latent_dim, network_architecture=experiment_architecture)
        self.models_dict = {'deployable': {'inference_model': None, 'generative_model': None},
                            'trainable': {'vae_model': None}}
        # init the base class' inputs and deployable models and reuse them in the paer
        super(GaussianVariationalAutoencoder, self).__init__(data_dim=data_dim, noise_dim=latent_dim,
                                                             latent_dim=latent_dim, name_prefix='vae',
                                                             resume_from=resume_from,
                                                             deployable_models_only=deployable_models_only)
        if resume_from is None:
            posterior_approximation, latent_mean, latent_log_var = self.encoder(self.data_input, is_learning=True)
            reconstruction_log_likelihood = self.decoder([self.data_input, posterior_approximation], is_learning=True)
            vae_loss = VAELossLayer(name='vae_loss')([reconstruction_log_likelihood, latent_mean, latent_log_var])
            self.vae_model = Model(inputs=self.data_input, outputs=vae_loss)
            self.vae_model.compile(optimizer=RMSprop(lr=1e-3), loss=None)

        self.models_dict['trainable']['vae_model'] = self.vae_model
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
