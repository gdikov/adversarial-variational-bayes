from keras.models import Model
from keras.optimizers import Adam
from networks.discriminator import Discriminator
from networks.encoder import Encoder
from networks.decoder import Decoder
from keras.layers import Input
from edward.models import MultivariateNormalDiag
import keras.backend as K


class AdversarialVariationalBayes(object):
    def __init__(self, data_dim, latent_dim, noise_dim=1, batch_size=32, with_ac=False, mode='generative'):

        # define the encoder input as the concatenated data and noise variable (renormalisation trick)
        data_input = Input(batch_shape=(batch_size, data_dim), name='x')
        noise_input = MultivariateNormalDiag(loc=K.zeros((batch_size, noise_dim)),
                                             scale_diag=K.ones((batch_size, noise_dim)))
        encoder_input = K.concatenate([data_input, noise_input], axis=1)
        self.encoder = Encoder(inputs=encoder_input, output_shape=latent_dim)

        # define a standard Normal prior distribution of the latent variables
        prior = MultivariateNormalDiag(loc=K.zeros((batch_size, latent_dim)),
                                       scale_diag=K.ones((batch_size, latent_dim)))

        self.discriminator = Discriminator(inputs={'x': data_input, 'q(z|x)': self.encoder.get_output(), 'p(z)': prior},
                                           output_shape=batch_size)

        self.decoder = Decoder(inputs={'q(z|x)': self.encoder.get_output(), 'p(z)': prior}, output_shape=data_dim)
        self._model = self._build()

    def _build(self):
        optimiser = Adam()
        summarised_loss = [self.discriminator.get_loss(),
                           self.decoder.get_loss(),
                           self.encoder.get_loss()]
        inputs = [self.encoder.get_input(), self.discriminator.get_input()]
        predictions = [self.decoder.get_output()]
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimiser, loss=summarised_loss, metrics=['accuracy'])
        return model

    def fit(self, data, target):
        train_history = self._model.fit(x=data, y=target)
        return train_history

    def evaluate(self):
        # self._model.evaluate()
        pass