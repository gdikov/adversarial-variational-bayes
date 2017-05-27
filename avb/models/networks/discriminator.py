from keras.layers import Dense, Dot, Activation, Concatenate, BatchNormalization
from keras.models import Model, Input


class Discriminator(object):
    def __init__(self, data_dim, latent_dim):

        discriminator_input_data = Input(shape=(data_dim,), name='disc_input_data')
        discriminator_input_latent = Input(shape=(latent_dim,), name='disc_input_latent')

        discriminator_body_data = Dense(512, activation='relu', name='disc_body1.1')(discriminator_input_data)
        discriminator_body_data = Dense(512, activation='relu', name='disc_body1.2')(discriminator_body_data)

        discriminator_body_latent = Dense(512, activation='relu', name='disc_body2.1')(discriminator_input_latent)
        discriminator_body_latent = Dense(512, activation='relu', name='disc_body2.2')(discriminator_body_latent)

        merged_data_latent = Dot(axes=1, name='disc_merge')([discriminator_body_data, discriminator_body_latent])
        discriminator_output = Activation(activation='sigmoid', name='disc_output')(merged_data_latent)



        # merged_data_latent = Concatenate(axis=1, name='disc_merge')([discriminator_input_data, discriminator_input_latent])
        #
        # discriminator_body = Dense(256, activation='relu', name='disc_body1')(merged_data_latent)
        # discriminator_body = Dense(256, activation='relu', name='disc_body2')(discriminator_body)
        #
        # discriminator_output = Dense(1, activation='sigmoid', name='disc_out')(discriminator_body)

        self.discriminator_model = Model(inputs=[discriminator_input_data, discriminator_input_latent],
                                         outputs=discriminator_output,
                                         name='discriminator')

    def __call__(self, *args, **kwargs):
        return self.discriminator_model(args[0])
