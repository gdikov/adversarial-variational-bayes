import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataIterator(object):
    def __init__(self, seed, data_dim, latent_dim, prior_distribution='standard_normal'):
        np.random.seed(seed)
        self.data_dim = data_dim
        self.latent_dim = latent_dim

    @staticmethod
    def adjust_data_size(data, batch_size):
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
        altered_data, n_batches = self.adjust_data_size(data, batch_size=batch_size)
        iterator = getattr(self, 'iter_data_{}'.format(mode))
        return iterator(altered_data, n_batches, **kwargs), n_batches


class AVBDataIterator(DataIterator):
    def __init__(self, data_dim, latent_dim, seed=7, prior_distribution='standard_normal'):
        super(AVBDataIterator, self).__init__(seed=seed, data_dim=data_dim, latent_dim=latent_dim,
                                              prior_distribution=prior_distribution)

    def iter_data_training(self, data, n_batches, **kwargs):
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            batches_indices = np.split(indices_new_order, n_batches)
            # run for 1 epoch
            for batch_indices in batches_indices:
                yield data[batch_indices].astype(np.float32)

    def iter_data_inference(self, data, n_batches, **kwargs):
        data_size = data.shape[0]
        while True:
            for batch_indices in np.split(np.arange(data_size), n_batches):
                yield data[batch_indices].astype(np.float32)

    def iter_data_generation(self, data, n_batches, **kwargs):
        data_size = data.shape[0]
        while True:
            for batch_indices in np.split(np.arange(data_size), n_batches):
                yield data[batch_indices].astype(np.float32)

VAEDataIterator = AVBDataIterator
