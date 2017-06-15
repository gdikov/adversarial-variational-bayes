from numpy import save as save_array
from os.path import join as path_join

from avb.utils.visualisation import plot_latent_2d, plot_sampled_data, plot_reconstructed_data
from avb.model_trainer import AVBModelTrainer, VAEModelTrainer
from avb.utils.datasets import load_npoints, load_mnist
from avb.utils.logger import logger

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def run_synthetic_experiment(model='vae'):
    logger.info("Starting an experiment on the synthetic dataset using {} model".format(model))
    data_dim = 4
    data = load_npoints(n=data_dim)

    train_data, train_labels = data['data'], data['target']

    if model == 'vae':
        trainer = VAEModelTrainer(data_dim=data_dim, latent_dim=2, experiment_name='synthetic', overwrite=True)
    elif model == 'avb':
        trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim, experiment_name='synthetic',
                                  overwrite=True, use_adaptive_contrast=False, optimiser_params={'lr': 1e-3})
    else:
        raise ValueError('Unknown model type. Supported models: `vae` and `avb`.')

    model_dir = trainer.run_training(train_data, batch_size=1024, epochs=2000)
    trained_model = trainer.get_model()

    reconstructions = trained_model.reconstruct(train_data, batch_size=1024)
    save_array(path_join(model_dir, 'reconstructed_samples.npy'), reconstructions)
    plot_reconstructed_data(train_data[:100], reconstructions[:100], fig_dirpath=model_dir)
    latent_vars = trained_model.infer(train_data, batch_size=1024)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    plot_latent_2d(latent_vars, train_labels, fig_dirpath=model_dir)
    generations = trained_model.generate(n_samples=100, batch_size=100)
    save_array(path_join(model_dir, 'generated_samples.npy'), generations)
    plot_sampled_data(generations, fig_dirpath=model_dir)
    return model_dir


def run_mnist_experiment(model='vae'):
    logger.info("Starting an experiment on the MNIST dataset using {} model".format(model))
    data_dim = 28**2
    latent_dim = 8
    data = load_mnist(binarised=True, one_hot=False)

    train_data, train_labels = data['data'], data['target']

    if model == 'vae':
        trainer = VAEModelTrainer(data_dim=data_dim, latent_dim=latent_dim, experiment_name='mnist', overwrite=True)
    elif model == 'avb':
        trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=latent_dim, noise_dim=data_dim / 28,
                                  experiment_name='mnist', overwrite=True, use_adaptive_contrast=False)
    elif model == 'avb+ac':
        trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=latent_dim, noise_dim=data_dim / 28,
                                  experiment_name='mnist', overwrite=True, use_adaptive_contrast=True)
    else:
        raise ValueError('Unknown model type. Supported models: `vae`, `avb` and `avb+ac`.')

    model_dir = trainer.run_training(train_data, batch_size=500, epochs=300)
    trained_model = trainer.get_model()

    reconstructions = trained_model.reconstruct(train_data, batch_size=1000)
    save_array(path_join(model_dir, 'reconstructed_samples.npy'), reconstructions)
    plot_reconstructed_data(train_data[:100], reconstructions[:100], fig_dirpath=model_dir)
    latent_vars = trained_model.infer(train_data, batch_size=1000)
    save_array(path_join(model_dir, 'latent_samples.npy'), latent_vars)
    if latent_dim == 2:
        plot_latent_2d(latent_vars, train_labels, fig_dirpath=model_dir)
    generations = trained_model.generate(n_samples=100, batch_size=100)
    save_array(path_join(model_dir, 'generated_samples.npy'), generations)
    plot_sampled_data(generations, fig_dirpath=model_dir)
    return model_dir


if __name__ == '__main__':
    run_synthetic_experiment('avb')
