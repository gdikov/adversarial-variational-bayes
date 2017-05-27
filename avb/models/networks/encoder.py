from keras.layers import Concatenate, Dense, Input
from keras.models import Model


class Encoder(object):
    def __init__(self, data_dim, noise_dim, latent_dim):
        data_input = Input(shape=(data_dim,), name='enc_input_data')
        noise_input = Input(shape=(noise_dim,), name='enc_input_noise')
        encoder_input = Concatenate(axis=1, name='enc_data_noise_concat')([data_input, noise_input])

        encoder_body = Dense(512, activation='relu', name='enc_body1')(encoder_input)
        encoder_body = Dense(512, activation='relu', name='enc_body2')(encoder_body)

        latent_factors = Dense(latent_dim, activation=None, name='enc_latent')(encoder_body)

        self.encoder_model = Model(inputs=[data_input, noise_input], outputs=latent_factors, name='encoder')

    def __call__(self, *args, **kwargs):
        return self.encoder_model(args[0])
