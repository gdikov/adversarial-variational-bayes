import os
import logging
import shutil
from datetime import datetime
from numpy import argmin, savez, asscalar
from utils.config import load_config

from avb.models import AdversarialVariationalBayes, GaussianVariationalAutoencoder

config = load_config('global_config.yaml')
logger = logging.getLogger(__name__)


class ModelTrainer(object):
    """
    ModelTrainer is a wrapper around the AVBModel and VAEModel to train it, log and store the resulting models.
    """
    def __init__(self, model, experiment_name, overwrite=True, checkpoint_best=False):
        """
        Args:
            model: the model object to be trained  
            experiment_name: str, the name of the experiment/training used for logging purposes
            overwrite: bool, whether the trained model should overwrite existing one with the same experiment_name 
        """
        self.model = model
        self.overwrite = overwrite
        self.experiment_name = experiment_name
        self.model_dirname = os.path.join(config['general']['models_dir'], self.experiment_name)
        self.checkpoint_best = checkpoint_best

    def get_model(self):
        """
        Return the model which can be used for evaluation. 
        
        Returns:
            The model instance. 
        """
        return self.model

    @staticmethod
    def prepare_training():
        """
        Prepare dirs and info for training, clean old cache.
        
        Returns:
            Formatted training start timestamp.
        """
        if os.path.exists(config['general']['temp_dir']):
            shutil.rmtree(config['general']['temp_dir'])
        os.makedirs(config['general']['temp_dir'])
        training_starttime = datetime.now().isoformat()
        return training_starttime

    def finalise_training(self, training_starttime, loss_history=None):
        """
        Clean up and store best model after training.
        
        Args:
            training_starttime: str, the formatted starting timestamp
            loss_history: dict, of the loss history for each model loss layer
            
        Returns:
            In-place method.
        """
        if self.overwrite:
            model_dirname = os.path.join(config['general']['models_dir'], self.experiment_name)
        else:
            model_dirname = os.path.join(config['general']['models_dir'],
                                         self.experiment_name + '_{}'.format(datetime.now().isoformat()))
        try:
            if not os.path.exists(model_dirname):
                os.makedirs(model_dirname)

            checkpoints = [fname for fname in os.listdir(config['general']['temp_dir']) if 'interrupted' not in fname]
            if self.checkpoint_best:
                best_checkpoint = checkpoints[asscalar(argmin([float(fname.split('_')[3]) for fname in checkpoints]))]
                tmp_model_dirname = os.path.join(config['general']['temp_dir'], best_checkpoint)
                for f in os.listdir(tmp_model_dirname):
                    if f.endswith('.h5'):
                        shutil.move(os.path.join(tmp_model_dirname, f), os.path.join(model_dirname, f))
        except IOError:
            logger.error("Saving model in model directory failed miserably.")
        try:
            # save some meta info related to the training and experiment:
            with open(os.path.join(model_dirname, 'meta.txt'), 'w') as f:
                f.write('Training on {} started on {} and finished on {}'.format(self.experiment_name,
                                                                                 training_starttime,
                                                                                 datetime.now().isoformat()))
            if loss_history is not None:
                savez(os.path.join(model_dirname, 'loss.npz'), **loss_history)
        except IOError:
            logger.error("Saving train history and other meta-information failed.")

        return model_dirname

    def run_training(self, data, batch_size=32, epochs=1, save_interrupted=False):
        """
        Run training of the model on training data.
        
        Args:
            data: ndarray, the data array of shape (N, data_dim) 
            batch_size: int, the number of samples for one pass
            epochs: int, the number of whole data iterations for training
            save_interrupted: bool, whether the model should be dumped on KeyboardInterrup signal

        Returns:
            The folder name of trained the model.
        """
        training_starttime = self.prepare_training()
        loss_history = None
        try:
            loss_history = self.fit_model(data, batch_size, epochs)
            endmodel_dir = os.path.join(self.model_dirname, 'final')
            self.model.save(endmodel_dir, deployable_models_only=False, save_metainfo=True)
        except KeyboardInterrupt:
            if save_interrupted:
                interrupted_dir = os.path.join(self.model_dirname, 'interrupted_{}'.format(datetime.now().isoformat()))
                self.model.save(interrupted_dir, deployable_models_only=False, save_metainfo=True)
                logger.warning("Training has been interrupted and the models "
                               "have been dumped in {}. Exiting program.".format(interrupted_dir))
            else:
                logger.warning("Training has been interrupted.")
        finally:
            model_dirname = self.finalise_training(training_starttime, loss_history)
            return model_dirname

    def fit_model(self, data, batch_size, epochs):
        """
        Fit particular model to the training data. Should be implemented by each model separately.
        
        Args:
            data: ndarray, the training data array
            batch_size: int, the number of samples for one iteration
            epochs: int, number of whole data iterations per training

        Returns:
            Model history dict object with the training losses.
        """
        return None


class AVBModelTrainer(ModelTrainer):
    """
    ModelTrainer instance for the AVBModel.
    """
    def __init__(self, data_dim, latent_dim, noise_dim, experiment_name, schedule=None,
                 resume_from=None, overwrite=True, use_adaptive_contrast=False):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            noise_dim: int, flattened noise dimensionality
            experiment_name: str, name of the training/experiment for logging purposes
            schedule: dict, schedule of training discriminator and encoder-decoder networks
            resume_from: str, model directory containing pre-trained model from which the training should be resumed
            overwrite: bool, whether to overwrite the existing trained model with the same experiment_name
            use_adaptive_contrast: bool, whether to train according to the Adaptive Contrast algorithm
        """
        avb = AdversarialVariationalBayes(data_dim=data_dim, latent_dim=latent_dim, noise_dim=noise_dim,
                                          resume_from=resume_from, deployable_models_only=False,
                                          experiment_architecture=experiment_name,
                                          use_adaptive_contrast=use_adaptive_contrast)
        self.schedule = schedule or {'iter_discr': 1, 'iter_encdec': 1}
        super(AVBModelTrainer, self).__init__(model=avb, experiment_name=experiment_name, overwrite=overwrite)

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit the AVBModel to the training data.
        
        Args:
            data: ndarray, training data
            batch_size: int, batch size
            epochs: int, number of epochs
        
        Keyword Args:

        Returns:
            A loss history dict with discriminator and encoder-decoder losses.
        """
        loss_hisotry = self.model.fit(data, batch_size, epochs=epochs,
                                      discriminator_repetitions=self.schedule['iter_discr'])
        return loss_hisotry


class VAEModelTrainer(ModelTrainer):
    """
    ModelTrainer instance for the GaussianVariationalAutoencoder (as per [TODO: add citation to Kingma, Welling]).
    """

    def __init__(self, data_dim, latent_dim, experiment_name, overwrite=True):
        """
        Args:
            data_dim: int, flattened data dimensionality 
            latent_dim: int, flattened latent dimensionality
            experiment_name: str, name of the training/experiment for logging purposes
            overwrite: bool, whether to overwrite the existing trained model with the same experiment_name
        """
        vae = GaussianVariationalAutoencoder(data_dim=data_dim, latent_dim=latent_dim,
                                             experiment_architecture=experiment_name)
        super(VAEModelTrainer, self).__init__(model=vae, experiment_name=experiment_name, overwrite=overwrite)

    def fit_model(self, data, batch_size, epochs, **kwargs):
        """
        Fit the GaussianVAE to the training data.

        Args:
            data: ndarray, training data
            batch_size: int, batch size
            epochs: int, number of epochs

        Keyword Args:

        Returns:
            A loss history dict with the encoder-decoder loss.
        """
        loss_hisotry = self.model.fit(data, batch_size, epochs=epochs)
        return loss_hisotry
