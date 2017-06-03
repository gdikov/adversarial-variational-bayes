import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataIterator(object):
    def __init__(self, data, batch_size, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        data_size = data.shape[0]
        n_batches = data_size / float(batch_size)

        if n_batches - int(n_batches) > 0:
            additional_samples = int(np.ceil(n_batches) * batch_size - data_size)
            logger.warning("The dataset cannot be iterated in epochs completely with the current batch size "
                           "and it will be automatically augmented with {} more samples.".format(additional_samples))
            additional_samples_indices = np.random.choice(data_size, size=additional_samples)
            self.data = np.concatenate([data, data[additional_samples_indices]], axis=0)
        else:
            self.data = data
        self.n_batches = self.data.shape[0] // batch_size


class AVBDataIterator(DataIterator):
    def __init__(self, data, batch_size, shuffle=True, seed=None, noise_distribution='normal',
                 input_noise_dim=None, prior_noise_dim=None):
        super(AVBDataIterator, self).__init__(data, batch_size, shuffle=shuffle)

        self.input_noise_dim = input_noise_dim or data.shape[1]
        self.prior_noise_dim = prior_noise_dim
        np.random.seed(seed)
        if noise_distribution == 'normal':
            self.noise_sampler = np.random.standard_normal
        elif noise_distribution == 'uniform':
            self.noise_sampler = np.random.uniform
        else:
            raise ValueError("Unsupported noise distribution type: {}".format(noise_distribution))

    def iter_data_training(self):
        if self.prior_noise_dim is None:
            raise ValueError("Prior noise dimensionality must be initialised when iterating in training mode.")

        while True:
            indices_new_order = np.arange(self.data.shape[0])
            if self.shuffle:
                np.random.shuffle(indices_new_order)
            batches_indices = np.split(indices_new_order, self.n_batches)
            # run for 1 epoch
            for batch_indices in batches_indices:
                random_noise_data = self.noise_sampler(size=(self.batch_size, self.input_noise_dim))
                random_noise_prior = self.noise_sampler(size=(self.batch_size, self.prior_noise_dim))
                yield [self.data[batch_indices], random_noise_data, random_noise_prior]

    def iter_data_inference(self):
        if self.shuffle:
            logger.warning("Shuffling while iterating in inference mode doesn't make sense and will be skipped.")
        while True:
            for batch_indices in np.split(np.arange(self.data.shape[0]), self.n_batches):
                random_noise_data = self.noise_sampler(size=(self.batch_size, self.input_noise_dim))
                yield [self.data[batch_indices], random_noise_data]

    def iter_data_generation(self):
        if self.shuffle:
            logger.warning("Shuffling while iterating in generation mode doesn't make sense and will be skipped.")
        while True:
            for batch_indices in np.split(np.arange(self.data.shape[0]), self.n_batches):
                yield self.data[batch_indices]


def iter_data(data=None, batch_size=32, mode='training', **kwargs):
    seed = kwargs.get('seed', 7)
    input_noise_dim = kwargs.get('input_noise_dim', data.shape[1])
    if mode == 'inference':
        data_iterator = AVBDataIterator(data=data, batch_size=batch_size, shuffle=False, seed=seed,
                                        noise_distribution='normal', input_noise_dim=input_noise_dim)

        return data_iterator.iter_data_inference(), data_iterator.n_batches
    elif mode == 'generation':
        data_iterator = AVBDataIterator(data=data, batch_size=batch_size, shuffle=False)
        return data_iterator.iter_data_generation(), data_iterator.n_batches
    elif mode == 'training':
        latent_dim = kwargs.get('latent_dim')
        data_iterator = AVBDataIterator(data=data, batch_size=batch_size, shuffle=True,
                                        seed=seed, noise_distribution='normal',
                                        input_noise_dim=input_noise_dim, prior_noise_dim=latent_dim)
        return data_iterator.iter_data_training(), data_iterator.n_batches
