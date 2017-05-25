from keras.layers import Dense
from keras.models import Model, Input


class Discriminator(object):
    def __init__(self, data_dim, latent_dim):

        discriminator_input = Input(shape=(data_dim + latent_dim,), name='discriminator_input')

        discriminator_body = Dense(256, activation='relu')(discriminator_input)
        discriminator_body = Dense(256, activation='relu')(discriminator_body)

        class_probability = Dense(1, activation='sigmoid', name='discriminator_output')(discriminator_body)

        self.discriminator_model = Model(inputs=discriminator_input, outputs=class_probability, name='Discriminator')

    def __call__(self, *args, **kwargs):
        return self.discriminator_model(args[0])
