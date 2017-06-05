from avb.model_trainer import AVBModelTrainer
from avb.models.avb import AdversarialVariationalBayes
from avb.models.vanilla_vae import GaussianVariationalAutoencoder
from utils.datasets import load_npoints, load_mnist
from utils.visualisation import plot_latent_2d, plot_sampled_data

# Generative:
#
#   * 4-points syntetic
#   * MNIST
#   * CelebA

if __name__ == '__main__':
    data_dim = 4
    data = load_npoints(n=data_dim)
    # data_dim = 28*28
    # data = load_mnist(binarised=True, one_hot=False)

    train_data, train_labels = data['data'], data['target']

    vae = GaussianVariationalAutoencoder(data_dim=data_dim, latent_dim=2)
    vae.fit(train_data, batch_size=1000, epochs=10)

    # trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim, experiment_name='4points',
    #                           overwrite=True)
    # model_dir = trainer.run_training(train_data, batch_size=1024, epochs=10)
    trained_model = vae#trainer.get_model()

    latent_vars = trained_model.infer(train_data, batch_size=512)
    plot_latent_2d(latent_vars, train_labels, fig_dirpath='data/')

    generations = trained_model.generate(n_samples=100, batch_size=100)
    plot_sampled_data(generations, sample_side_size=2, fig_dirpath='data/')
