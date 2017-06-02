import numpy as np
import os
from tqdm import tqdm

from keras.models import Model, Input, model_from_json
from keras.optimizers import Adam as Optimiser
from freezable_model import FreezableModel

from networks import Encoder, Decoder, Discriminator
from .avb_loss import DiscriminatorLossLayer, DecoderLossLayer
from ..data_iterator import iter_data
from utils.config import load_config

config = load_config('global_config.yaml')
np.random.seed(config['general']['seed'])


class AdversarialVariationalBayes(object):
    def __init__(self, data_dim, latent_dim=2, noise_dim=None, restore_models=None, deployable_models_only=False):

        self.data_dim = data_dim
        self.noise_dim = noise_dim or data_dim
        self.latent_dim = latent_dim

        if restore_models is not None:
            restorable_models = os.listdir(restore_models)
            if not deployable_models_only:
                if 'trainable_encoderdecoder_model.h5' in restorable_models and \
                                'trainable_discriminator_model.h5' in restorable_models:
                    with open('trainable_encoderdecoder_model.json', 'r') as j:
                        self._avb_dec_train = model_from_json(j.read())
                    self._avb_dec_train.load_weights(os.path.join(restore_models, 'trainable_encoderdecoder_model.h5'))
                    with open('trainable_discriminator_model.json', 'r') as j:
                        self._avb_disc_train = model_from_json(j.read())
                    self._avb_disc_train.load_weights(os.path.join(restore_models, 'trainable_discriminator_model.h5'))
                else:
                    ValueError("Could not find `trainable_encoderdecoder_model.h5` "
                               "and `trainable_discriminator_model.h5` in {}".format(restorable_models))
            if 'inference_model.h5' in restorable_models and \
               'generative_model.h5' in restorable_models:
                    with open('inference_model.json', 'r') as j:
                        self._inference_model = model_from_json(j.read())
                    self._inference_model.load_weights(os.path.join(restore_models, 'inference_model.h5'))
                    with open('generative_model.json', 'r') as j:
                        self._generative_model = model_from_json(j.read())
                    self._generative_model.load_weights(os.path.join(restore_models, 'generative_model.h5'))
            else:
                ValueError("Could not find `inference_model.h5` "
                           "and `generative_model.h5` in {}".format(restorable_models))
        else:
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

            # define the trainable models
            self._avb_disc_train = FreezableModel(inputs=[data_input, noise_input, prior_input],
                                                  outputs=discriminator_loss, name_prefix=['disc'])
            self._avb_dec_train = FreezableModel(inputs=[data_input, noise_input],
                                                 outputs=decoder_loss, name_prefix=['dec', 'enc'])

            self._avb_disc_train.freeze()
            self._avb_dec_train.unfreeze()
            self._avb_dec_train.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None)

            self._avb_disc_train.unfreeze()
            self._avb_dec_train.freeze()
            self._avb_disc_train.compile(optimizer=Optimiser(lr=1e-3, beta_1=0.5), loss=None)

            # define the deployable model (evaluation only)
            self._inference_model = Model(inputs=[data_input, noise_input], outputs=posterior_approximation)
            self._generative_model = Model(inputs=prior_input, outputs=decoder(prior_input, is_learning=False))

    def fit(self, data, batch_size=32, epochs=1, discriminator_repetitions=1):
        data_iterator, iters_per_epoch = iter_data(data, batch_size, mode='training', seed=config['general']['seed'],
                                                   latent_dim=self.latent_dim, input_noise_dim=self.noise_dim)

        epoch_loss = np.inf
        for ep in tqdm(xrange(epochs)):
            epoch_loss_history_encdec = []
            epoch_loss_history_disc = []
            for it in xrange(iters_per_epoch):
                data_batch = data_iterator.next()
                loss_autoencoder = self._avb_dec_train.train_on_batch(data_batch[:-1], None)
                epoch_loss_history_encdec.append(loss_autoencoder)
                for _ in xrange(discriminator_repetitions):
                    loss_discriminator = self._avb_disc_train.train_on_batch(data_batch, None)
                    epoch_loss_history_disc.append(loss_discriminator)

            current_epoch_loss = np.mean(epoch_loss_history_encdec) + np.mean(epoch_loss_history_disc)
            if current_epoch_loss < 0.9 * epoch_loss:
                epoch_loss = current_epoch_loss
                self.save(os.path.join(config['general']['temp_dir'], 'ep_{}_loss_{}'.format(ep, epoch_loss)),
                          deployable_models_only=False)

    def infer(self, data, batch_size=32):
        data_iterator, n_iters = iter_data(data, batch_size, mode='inference', seed=config['general']['seed'],
                                           input_noise_dim=self.noise_dim)
        latent_samples = self._inference_model.predict_generator(data_iterator, steps=n_iters)
        return latent_samples

    def generate(self, n_samples, batch_size=32):
        n_samples_per_axis = complex(int(np.sqrt(n_samples)))
        data = np.mgrid[-300:100:n_samples_per_axis, -50:300:n_samples_per_axis].reshape(2, -1).T
        data_iterator, n_iters = iter_data(data, batch_size=batch_size, mode='generation',
                                           seed=config['general']['seed'], latent_dim=self.latent_dim)
        data_probs = self._generative_model.predict_generator(data_iterator, steps=n_iters)
        sampled_data = np.random.binomial(1, p=data_probs)
        return sampled_data

    def save(self, dirname, deployable_models_only=False, save_metainfo=False):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if not deployable_models_only:
            self._avb_dec_train.save(os.path.join(dirname, 'trainable_encoderdecoder_model.h5'), include_optimizer=True)
            self._avb_disc_train.save(os.path.join(dirname, 'trainable_discriminator_model.h5'), include_optimizer=True)
        self._inference_model.save_weights(os.path.join(dirname, 'inference_model.h5'))
        self._generative_model.save_weights(os.path.join(dirname, 'generative_model.h5'))

        if save_metainfo:
            with open(os.path.join(dirname, 'inference_model.json'), 'w') as f:
                f.write(self._inference_model.to_json())
            with open(os.path.join(dirname, 'generative_model.json'), 'w') as f:
                f.write(self._generative_model.to_json())
            with open(os.path.join(dirname, 'trainable_encoderdecoder_model.json'), 'w'):
                f.write(self._avb_dec_train.to_json())
            with open(os.path.join(dirname, 'trainable_discriminator_model.json'), 'w'):
                f.write(self._avb_disc_train.to_json())
            from keras.utils import plot_model
            plot_model(self._avb_disc_train, to_file='discriminator_model.png')
            plot_model(self._avb_dec_train, to_file='encoderdecoder_model.png')
