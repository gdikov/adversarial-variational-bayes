from avb.model_trainer import AVBModelTrainer, VAEModelTrainer
from utils.datasets import load_npoints, load_mnist
from utils.visualisation import plot_latent_2d, plot_sampled_data
from utils.logger import logger


def run_synthetic_experiment(model='vae'):
    data_dim = 4
    data = load_npoints(n=data_dim)

    train_data, train_labels = data['data'], data['target']

    if model == 'vae':
        trainer = VAEModelTrainer(data_dim=data_dim, latent_dim=2, experiment_name='synthetic', overwrite=True)
    elif model == 'avb':
        trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim, experiment_name='synthetic',
                                  overwrite=True, use_adaptive_contrast=False)
    else:
        raise ValueError('Unknown model type. Supported models: `vae` and `avb`.')

    model_dir = trainer.run_training(train_data, batch_size=1024, epochs=2000)
    trained_model = trainer.get_model()
    latent_vars = trained_model.infer(train_data, batch_size=1000)
    plot_latent_2d(latent_vars, train_labels, fig_dirpath=model_dir)
    generations = trained_model.generate(n_samples=100, batch_size=100)
    plot_sampled_data(generations, sample_side_size=2, fig_dirpath=model_dir)
    return model_dir


def run_mnist_experiment(model='vae'):
    data_dim = 28
    data = load_mnist(binarised=True, one_hot=False)

    train_data, train_labels = data['data'], data['target']

    if model == 'vae':
        trainer = VAEModelTrainer(data_dim=data_dim, latent_dim=2, experiment_name='mnist', overwrite=True)
    elif model == 'avb':
        trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim / 28, experiment_name='mnist',
                                  overwrite=True, use_adaptive_contrast=False)
    elif model == 'avb+ac':
        trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim / 28, experiment_name='mnist',
                                  overwrite=True, use_adaptive_contrast=True)
    else:
        raise ValueError('Unknown model type. Supported models: `vae`, `avb` and `avb+ac`.')

    model_dir = trainer.run_training(train_data, batch_size=500, epochs=1000)
    trained_model = trainer.get_model()
    latent_vars = trained_model.infer(train_data, batch_size=1000)
    plot_latent_2d(latent_vars, train_labels, fig_dirpath=model_dir)
    generations = trained_model.generate(n_samples=100, batch_size=100)
    plot_sampled_data(generations, sample_side_size=28, fig_dirpath=model_dir)
    return model_dir


if __name__ == '__main__':
    run_synthetic_experiment('avb')
