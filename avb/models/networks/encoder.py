from keras.layers import Concatenate, Dense, Input
from keras.models import Model


class Encoder(object):
    def __init__(self, data_dim, noise_dim, latent_dim):
        data_input = Input(shape=(data_dim,))
        noise_input = Input(shape=(noise_dim,))
        encoder_input = Concatenate(axis=1, name='data_noise_concat')([data_input, noise_input])

        encoder_body = Dense(512, activation='relu')(encoder_input)
        encoder_body = Dense(512, activation='relu')(encoder_body)

        latent_factors = Dense(latent_dim, activation=None)(encoder_body)

        self.encoder_model = Model(inputs=[data_input, noise_input], outputs=latent_factors, name='Encoder')

    def __call__(self, *args, **kwargs):
        return self.encoder_model(args[0])
