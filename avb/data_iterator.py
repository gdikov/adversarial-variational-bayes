import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataIterator(object):
    def __init__(self, seed, data_dim, latent_dim, noise_dim=None, noise_distribution='normal'):
        np.random.seed(seed)
        self.data_dim = data_dim
        self.noise_dim = noise_dim or latent_dim
        self.latent_dim = latent_dim

        if noise_distribution == 'normal':
            self.noise_sampler = np.random.standard_normal
        elif noise_distribution == 'uniform':
            self.noise_sampler = np.random.uniform
        else:
            raise ValueError("Unsupported noise distribution type: {}".format(noise_distribution))

    @staticmethod
    def tailor_data_size(data, batch_size):
        data_size = data.shape[0]
        n_batches = data_size / float(batch_size)

        if n_batches - int(n_batches) > 0:
            additional_samples = int(np.ceil(n_batches) * batch_size - data_size)
            logger.warning("The dataset cannot be iterated in epochs completely with the current batch size "
                           "and it will be automatically augmented with {} more samples.".format(additional_samples))
            additional_samples_indices = np.random.choice(data_size, size=additional_samples)
            altered_data = np.concatenate([data, data[additional_samples_indices]], axis=0)
        else:
            altered_data = data
        n_batches = altered_data.shape[0] // batch_size
        return altered_data, n_batches

    def iter_data_inference(self, data, n_batches, **kwargs):
        raise NotImplementedError

    def iter_data_generation(self, data, n_batches,  **kwargs):
        raise NotImplementedError

    def iter_data_training(self, data, n_batches,  **kwargs):
        raise NotImplementedError

    def iter(self, data, batch_size, mode='training', **kwargs):
        altered_data, n_batches = self.tailor_data_size(data, batch_size=batch_size)
        iterator = getattr(self, 'iter_data_{}'.format(mode))
        return iterator(altered_data, n_batches, **kwargs), n_batches


class AVBDataIterator(DataIterator):
    def __init__(self, data_dim, latent_dim, noise_dim, seed=7, noise_distribution='normal'):
        super(AVBDataIterator, self).__init__(seed=seed, data_dim=data_dim, latent_dim=latent_dim,
                                              noise_dim=noise_dim, noise_distribution=noise_distribution)

    def iter_data_training(self, data, n_batches, **kwargs):
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        batch_size = data_size // n_batches
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            batches_indices = np.split(indices_new_order, n_batches)
            # run for 1 epoch
            for batch_indices in batches_indices:
                random_noise_data = self.noise_sampler(size=(batch_size, self.noise_dim))
                random_noise_prior = self.noise_sampler(size=(batch_size, self.latent_dim))
                yield [data[batch_indices], random_noise_data, random_noise_prior]

    def iter_data_inference(self, data, n_batches, **kwargs):
        data_size = data.shape[0]
        batch_size = data_size // n_batches
        while True:
            for batch_indices in np.split(np.arange(data_size), n_batches):
                random_noise_data = self.noise_sampler(size=(batch_size, self.noise_dim))
                yield [data[batch_indices], random_noise_data]

    def iter_data_generation(self, data, n_batches, **kwargs):
        while True:
            for batch_indices in np.split(np.arange(data.shape[0]), n_batches):
                yield data[batch_indices]

VAEDataIterator = AVBDataIterator
