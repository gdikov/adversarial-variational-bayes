from __future__ import absolute_import
from six import iteritems

import numpy as np
import os

from keras.models import Model, Input, load_model
from scipy.stats import norm as standard_gaussian

from ..utils.config import load_config

config = load_config('global_config.yaml')
np.random.seed(config['seed'])


class BaseVariationalAutoencoder(object):
    """
    Base class for the Adversarial Variational Bayes Autoencoder and the vanilla Variational Autoencoder.
    """

    def __init__(self, data_dim, latent_dim=2, noise_dim=None, name_prefix=None):
        """
        Args:
            data_dim: int, flattened data dimensionality
            latent_dim: int, flattened latent dimensionality
            noise_dim: int, flattened noise, dimensionality
            name_prefix: str, the prefix of named layers
        """
        self.data_dim = data_dim
        self.noise_dim = noise_dim or latent_dim
        self.latent_dim = latent_dim
        name_prefix = name_prefix or 'base_vae'

        if not hasattr(self, 'encoder') and hasattr(self, 'decoder'):
            raise AttributeError("Initialise the attributes `encoder` and `decoder` in the child classes first!")

        self.data_input = Input(shape=(data_dim,), name='{}_data_input'.format(name_prefix))
        self.latent_input = Input(shape=(latent_dim,), name='{}_latent_prior_input'.format(name_prefix))

        # define the testing models
        self.inference_model = Model(inputs=self.data_input,
                                     outputs=self.encoder(self.data_input, is_learning=False))
        self.generative_model = Model(inputs=self.latent_input,
                                      outputs=self.decoder(self.latent_input, is_learning=False))

    def fit(self, data, batch_size=32, epochs=1, **kwargs):
        """
        Fit the model to the training data.

        Args:
            data: ndarray, data array of shape (N, data_dim)
            batch_size: int, the number of samples to be used at one training pass
            epochs: int, the number of epochs for training (whole size data iterations) 

        Returns:
            The training history as a dict of lists of the epoch-wise losses.
        """
        return None

    def infer(self, data, batch_size=32, **kwargs):
        """
        Infer latent factors given some data. 

        Args:
            data: ndarray, data array of shape (N, data_dim) 
            batch_size: int, the number of samples to be inferred at once

        Keyword Args:
            sampling_size: int, the number of noisy samples which will be inferred for a single data input sample

        Returns:
            The inferred latent factors as ndarray of shape (N, latent_dim) 
        """
        if not hasattr(self, 'data_iterator'):
            raise AttributeError("Initialise the data iterator in the child classes first!")
        sampling_size = kwargs.get('sampling_size', 1)
        data = np.repeat(data, sampling_size, axis=0)
        data_iterator, n_iters = self.data_iterator.iter(data, batch_size, mode='inference')
        latent_samples = self.inference_model.predict_generator(data_iterator, steps=n_iters)
        return latent_samples

    def generate(self, n_samples=100, batch_size=32, **kwargs):
        """
        Sample new data from the generator network.

        Args:
            n_samples: int, the number of samples to be generated 
            batch_size: int, number of generated samples are once

        Keyword Args:
            return_probs: bool, whether the output generations should be raw probabilities or sampled Bernoulli outcomes
            latent_samples: ndarray, alternative source of latent encoding, otherwise sampling will be applied

        Returns:
            The generated data as ndarray of shape (n_samples, data_dim)
        """
        return_probs = kwargs.get('return_probs', True)
        latent_samples = kwargs.get('latent_samples', None)

        if latent_samples is not None:
            data_iterator, n_iters = self.data_iterator.iter(latent_samples, batch_size=batch_size, mode='generation')
            data_probs = self.generative_model.predict_generator(data_iterator, steps=n_iters)
        else:
            if self.latent_dim == 2:
                # perform 2d grid search
                n_samples_per_axis = complex(int(np.sqrt(n_samples)))
                uniform_grid = np.mgrid[0.01:0.99:n_samples_per_axis, 0.01:0.99:n_samples_per_axis].reshape(2, -1).T
                latent_samples = standard_gaussian.ppf(uniform_grid)
            else:
                latent_samples = np.random.standard_normal(size=(n_samples, self.latent_dim))
            data_iterator, n_iters = self.data_iterator.iter(latent_samples, batch_size=batch_size, mode='generation')
            data_probs = self.generative_model.predict_generator(data_iterator, steps=n_iters)
        if return_probs:
            return data_probs
        sampled_data = np.random.binomial(1, p=data_probs)
        return sampled_data

    def reconstruct(self, data, batch_size=32, **kwargs):
        """
        Reconstruct input data from latent encoding. Used mainly for goodness evaluation purposes.

        Args:
            data: ndarray, input data of shape (N, data_dim) to be encoded and decoded
            batch_size: int, the number of samples to be computed at one pass

        Keyword Args:
            sampling_size: int, the number of noisy samples which will be reconstructed for a single data input sample

        Returns:
            A ndarray of the same shape as the input, representing the reconstructed samples
        """
        sampling_size = kwargs.get('sampling_size', 1)
        latent_samples = self.infer(data, batch_size, sampling_size=sampling_size)
        reconstructed_samples = self.generate(batch_size=batch_size, latent_samples=latent_samples, return_probs=True)
        return reconstructed_samples

    def save(self, dirname):
        """
        Save the weights of all AVB sub-models, so that training can be resumed. 

        Args:
            dirname: str, path to the folder to store the weights

        Returns:
            In-place method.
        """
        if not hasattr(self, 'models_dict'):
            raise AttributeError("Initialise the model dict in the child class first!")

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for name, model in iteritems(self.models_dict):
            model.save(os.path.join(dirname, '{}.h5'.format(name)), include_optimizer=True, overwrite=True)

    def load(self, dirname, custom_layers=None):
        """
        Load models from json and h5 files for testing or training.
        
        Args:
            dirname: str, the path to the models directory
            custom_layers: dict, custom Keras layers (e.g. loss layers)

        Returns:
            In-place method.
        """
        if not hasattr(self, 'models_dict'):
            raise AttributeError("Initialise the model dict in the child class first!")

        for name in self.models_dict.keys():
            # this weights loading is not elegant but it seems that Keras is initialising the weights before training
            # and the loaded model is becoming useless (maybe a Keras bug in saving/loading?). If fixed, this should be
            # refactored too.
            # NOTE: this is not loading the optimiser parameters
            existing_model = getattr(self, name)
            existing_model.load_weights(os.path.join(dirname, '{}.h5'.format(name)))
