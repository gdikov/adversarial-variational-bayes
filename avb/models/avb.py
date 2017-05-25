import numpy as np
from keras.models import Model
from keras.layers import Input

from networks import Encoder, Decoder, Discriminator
from .avb_loss import AVBLossLayer


class AdversarialVariationalBayes(object):
    def __init__(self, data_dim, latent_dim, noise_dim=4):

        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        # define the encoder input as the concatenated data and noise variable (renormalisation trick)
        data_input = Input(shape=(data_dim,), name='input_data')
        noise_input = Input(shape=(noise_dim,), name='input_noise')
        prior_input = Input(shape=(latent_dim,), name='input_latent_prior')

        self.encoder_network = Encoder(inputs=[data_input, noise_input], output_shape=latent_dim)
        encoder_output = self.encoder_network.get_output()
        self.decoder_network = Decoder(inputs=[encoder_output, data_input],
                                       output_shape=data_dim)
        self.discriminator_network = Discriminator(inputs=[data_input, encoder_output, prior_input],
                                                   output_shape=1)

        avb_inputs = [data_input, noise_input, prior_input]
        discr_output_posterior, discr_output_prior = self.discriminator_network.get_output()
        decoder_output = self.decoder_network.get_output()
        avb_output = AVBLossLayer()([discr_output_posterior, discr_output_prior, decoder_output])

        self.complete_avb_model = Model(inputs=avb_inputs, outputs=avb_output)
        self.complete_avb_model.compile(optimizer='adagrad', loss=None)

        # detach the encoder and decoder for inference and data generation purposes
        self.encoder_model = Model(inputs=[data_input, noise_input], outputs=encoder_output)
        # self.decoder_model = Model(inputs=prior_input, outputs=decoder_output)

    def _iter_data(self, data, batch_size, mode=None):
        data_size = data.shape[0]
        while True:
            indices = np.random.choice(data_size, size=batch_size, replace=True)
            random_noise_data = np.random.standard_normal(size=(batch_size, self.noise_dim))
            random_noise_prior = np.random.standard_normal(size=(batch_size, self.latent_dim))
            if mode == 'inference':
                yield [data[indices], random_noise_data], None
            yield [data[indices], random_noise_data, random_noise_prior], None

    def fit(self, data, batch_size=32):
        self.complete_avb_model.fit_generator(self._iter_data(data, batch_size),
                                              steps_per_epoch=32,
                                              epochs=100,
                                              workers=1)

    def infer(self, data, batch_size=32):
        latent_vars = self.encoder_model.predict_generator(self._iter_data(data, batch_size, mode='inference'), steps=1)
        return latent_vars

    def generate(self):
        pass
