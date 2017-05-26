from keras.layers import Dense, Dot, Activation
from keras.models import Model, Input


class Discriminator(object):
    def __init__(self, data_dim, latent_dim):

        discriminator_input_data = Input(shape=(data_dim,), name='discriminator_input_data')
        discriminator_input_latent = Input(shape=(latent_dim,), name='discriminator_input_latent')

        discriminator_body_data = Dense(512, activation='relu')(discriminator_input_data)
        discriminator_body_data = Dense(512, activation='relu')(discriminator_body_data)

        discriminator_body_latent = Dense(512, activation='relu')(discriminator_input_latent)
        discriminator_body_latent = Dense(512, activation='relu')(discriminator_body_latent)

        merged_data_latent = Dot(axes=-1)([discriminator_body_data, discriminator_body_latent])
        discriminator_output = Activation(activation='sigmoid', name='discriminator_output')(merged_data_latent)

        self.discriminator_model = Model(inputs=[discriminator_input_data, discriminator_input_latent],
                                         outputs=discriminator_output,
                                         name='Discriminator')

    def __call__(self, *args, **kwargs):
        return self.discriminator_model(args[0])
