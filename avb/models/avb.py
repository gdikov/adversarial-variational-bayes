import numpy as np
import os
from datetime import datetime
import dill
from tqdm import tqdm

from keras.models import Model, Input
from keras.optimizers import Adam as Optimiser
from freezable_model import FreezableModel

from networks import Encoder, Decoder, Discriminator
from .avb_loss import DiscriminatorLossLayer, DecoderLossLayer
from ..data_iterator import iter_data
from utils.config import load_config

config = load_config('../global_config.yaml')
np.random.seed(config['seed'])


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

        discriminator_loss = DiscriminatorLossLayer(name='disc_loss')([discriminator_output_prior,
                                                                       discriminator_output_posterior])
        decoder_loss = DecoderLossLayer(name='dec_loss')([reconstruction_log_likelihood,
                                                          discriminator_output_posterior])

        self._avb_disc_train = FreezableModel(inputs=[data_input, noise_input, prior_input],
                                              outputs=discriminator_loss, name_prefix=['disc'])
        self._avb_dec_train = FreezableModel(inputs=[data_input, noise_input, prior_input],
                                             outputs=decoder_loss, name_prefix=['dec', 'enc'])

        self._avb_disc_train.freeze()
        self._avb_dec_train.unfreeze()
        self._avb_dec_train.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None, metrics=['accuracy'])

        self._avb_disc_train.unfreeze()
        self._avb_dec_train.freeze()
        self._avb_disc_train.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None)

        self._inference_model = Model(inputs=[data_input, noise_input], outputs=posterior_approximation)
        self._generative_model = Model(inputs=prior_input, outputs=decoder(prior_input, is_learning=False))
        # from keras.utils import plot_model
        # plot_model(self._avb)

    def fit(self, data, batch_size=32, epochs=1, discriminator_repetitions=1):
        data_iterator, iters_per_epoch = iter_data(data, batch_size, mode='training', seed=config['seed'],
                                                   latent_dim=self.latent_dim, input_noise_dim=self.noise_dim)
        for ep in tqdm(xrange(epochs)):
            for it in xrange(iters_per_epoch):
                data_batch = data_iterator.next()
                self._avb_dec_train.train_on_batch(data_batch, None)
                for _ in xrange(discriminator_repetitions):
                    self._avb_disc_train.train_on_batch(data_batch, None)

    def infer(self, data, batch_size=32):
        data_iterator, n_iters = iter_data(data, batch_size, mode='inference', seed=config['seed'],
                                           input_noise_dim=self.noise_dim)
        latent_samples = self._inference_model.predict_generator(data_iterator, steps=n_iters)
        return latent_samples

    def generate(self, n_samples, batch_size=32):
        n_samples_per_axis = complex(int(np.sqrt(n_samples)))
        data = np.mgrid[-3:3:n_samples_per_axis, -3:3:n_samples_per_axis].reshape(2, -1).T
        data_iterator, n_iters = iter_data(data, batch_size=batch_size, mode='generation',
                                           seed=config['seed'], latent_dim=self.latent_dim)
        data_probs = self._generative_model.predict_generator(data_iterator, steps=n_iters)
        sampled_data = np.random.binomial(1, p=data_probs, size=data_probs.shape)
        return sampled_data

    def save(self, dirname, deployable_models_only=False):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if not deployable_models_only:
            timestamp = datetime.now().isoformat()
            path_to_trainable_models = os.path.join(config['temp_dir'], timestamp)
            if not os.path.exists(path_to_trainable_models):
                os.makedirs(path_to_trainable_models)
            # use cPickle to store all the information necessary to restart the training from that point
            with open(os.path.join(path_to_trainable_models, 'trainable_discriminator_model'), 'wb') as f:
                dill.dump(self._avb_disc_train, f)
            with open(os.path.join(path_to_trainable_models, 'trainable_decoder_model'), 'wb') as f:
                dill.dump(self._avb_dec_train, f)
        self._inference_model.save_weights(os.path.join(dirname, 'inference_model'))
        self._generative_model.save_weights(os.path.join(dirname, 'generative_model'))

    def load(self, model_paths, deployable_models_only=False):
        self._generative_model.load_weights(model_paths['generative_model'])
        self._inference_model.load_weights(model_paths['ingerence_model'])
        if not deployable_models_only:
            with open(model_paths['trainable_discriminator_model'], 'rb') as f:
                self._avb_disc_train = dill.load(f)
            with open(model_paths['trainable_decoder_model'], 'rb') as f:
                self._avb_dec_train = dill.load(f)
