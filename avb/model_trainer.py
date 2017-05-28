import logging

from utils.config import load_config

config = load_config('../global_config.yaml')
logger = logging.getLogger(__name__)


class AVBModelTrainer(object):
    def __init__(self, model):
        self.model = model

    def checkpoint_model(self):
        self.model.save(config['models_dir'])

    def prepare_training(self):
        pass

    def finalise_training(self):
        pass

    def start_training(self, data, batch_size=32, epochs=1, discriminator_repetitions=1):
        self.prepare_training()
        try:
            self.model.fit(data, batch_size, epochs=epochs, discriminator_repetitions=discriminator_repetitions)
        except KeyboardInterrupt:
            self.checkpoint_model()
            logger.warning("Training has been interrupted and the models "
                           "have been dumped in {}. Exiting program.".format(1))
        finally:
            self.finalise_training()
