from keras.models import Model
from keras.optimizers import Adam


class AdversarialVariationalBayes(object):
    def __init__(self, discriminator, encoder, decoder, with_ac=False):
        self.discriminator = discriminator
        self.encoder = encoder
        self.decoder = decoder
        self._model = self._build()

    def _build(self):
        optimiser = Adam()
        summarised_loss = [self.discriminator.get_loos(),
                           self.decoder.get_loss(),
                           self.encoder.get_loss()]
        inputs = [self.encoder.get_input(), self.discriminator.get_input()]
        predictions = [self.decoder.get_output()]
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimiser,
                      loss=summarised_loss,
                      metrics=['accuracy'])
        return model

    def fit(self, data, target):
        train_history = self._model.fit(x=data, y=target)
        return train_history

    def evaluate(self):
        # self._model.evaluate()
        pass