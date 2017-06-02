import os
import logging
import shutil
from datetime import datetime
from numpy import argmin
from utils.config import load_config

config = load_config('global_config.yaml')
logger = logging.getLogger(__name__)


class AVBModelTrainer(object):
    def __init__(self, model, training_name, overwrite=True):
        if isinstance(model, str) and model.endswith('.h5'):
            # interpret the model as path to the stored model and resume training
            pass
        self.model = model
        self.overwrite = overwrite
        self.training_name = training_name

    def prepare_training(self):
        if os.path.exists(config['general']['temp_dir']):
            shutil.rmtree(config['general']['temp_dir'])
        os.makedirs(config['general']['temp_dir'])
        self.training_starttime = datetime.now().isoformat()

    def finalise_training(self):
        checkpoints = [fname for fname in os.listdir(config['general']['temp_dir']) if 'interrupted' not in fname]
        best_checkpoint = checkpoints[argmin([float(fname.split('_')[3]) for fname in checkpoints])]
        if self.overwrite:
            model_dirname = os.path.join(config['general']['models_dir'], self.training_name)
        else:
            model_dirname = os.path.join(config['general']['models_dir'],
                                         self.training_name + '_{}'.format(datetime.now().isoformat()))
        if not os.path.exists(model_dirname):
            os.makedirs(model_dirname)
        tmp_model_dirname = os.path.join(config['general']['temp_dir'], best_checkpoint)
        for f in os.listdir(tmp_model_dirname):
            if f.endswith('.h5'):
                shutil.move(os.path.join(tmp_model_dirname, f), os.path.join(model_dirname, f))

        # save some meta info related to the training and experiment:
        with open(os.path.join(model_dirname, 'meta.txt'), 'w') as f:
            f.write('Training on {} started on {} and finished on {}'.format(self.training_name,
                                                                             self.training_starttime,
                                                                             datetime.now().isoformat()))
        return model_dirname

    def run_training(self, data, batch_size=32, epochs=1, schedule=None, save_interrupted=False):
        self.prepare_training()
        model_dirname = os.path.join(config['general']['models_dir'], self.training_name)
        try:
            schedule = schedule or {'iter_discr': 1, 'iter_encdec': 1}
            self.model.fit(data, batch_size, epochs=epochs, discriminator_repetitions=schedule['iter_discr'])
            model_dirname = os.path.join(model_dirname, 'end')
            self.model.save(model_dirname, deployable_models_only=False, save_metainfo=True)
        except KeyboardInterrupt:
            if save_interrupted:
                interrupted_dir = os.path.join(model_dirname, 'interrupted_{}'.format(datetime.now().isoformat()))
                self.model.save(interrupted_dir, deployable_models_only=False, save_metainfo=True)
                logger.warning("Training has been interrupted and the models "
                               "have been dumped in {}. Exiting program.".format(interrupted_dir))
            else:
                logger.warning("Training has been interrupted.")
        finally:
            model_dirname = self.finalise_training()
            return model_dirname
