from avb.model_trainer import AVBModelTrainer, VAEModelTrainer
from avb.models import AdversarialVariationalBayes, GaussianVariationalAutoencoder
from utils.datasets import load_npoints, load_mnist
from utils.visualisation import plot_latent_2d, plot_sampled_data

# Generative:
#
#   * 4-points syntetic
#   * MNIST
#   * CelebA

if __name__ == '__main__':
    # data_dim = 4
    # data = load_npoints(n=data_dim)
    data_dim = 28*28
    data = load_mnist(binarised=True, one_hot=False)

    train_data, train_labels = data['data'], data['target']

    # trainer = VAEModelTrainer(data_dim=data_dim, latent_dim=2, experiment_name='mnist', overwrite=True)
    # trainer.run_training(train_data, batch_size=1024, epochs=2000)

    trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim, experiment_name='mnist',
                              overwrite=True)
    model_dir = trainer.run_training(train_data, batch_size=1024, epochs=2000)
    trained_model = trainer.get_model()

    latent_vars = trained_model.infer(train_data, batch_size=10000)

    plot_latent_2d(latent_vars, train_labels, fig_dirpath='data/')

    generations = trained_model.generate(n_samples=100, batch_size=100)
    plot_sampled_data(generations, sample_side_size=2, fig_dirpath='data/')
