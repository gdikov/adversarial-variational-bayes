from avb.model_trainer import AVBModelTrainer
from avb.models.avb import AdversarialVariationalBayes
from avb.models.vae import VariationalAutoencoder
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

    vae = VariationalAutoencoder(data_dim=data_dim, latent_dim=2)

    # trainer = AVBModelTrainer(data_dim=data_dim, latent_dim=2, noise_dim=data_dim, experiment_name='4points', overwrite=True)
    # model_dir = trainer.run_training(train_data, batch_size=1024, epochs=10)

    # avb = trainer.get_model()
    # latent_vars = avb.infer(train_data, batch_size=512)
    # plot_latent_2d(latent_vars, train_labels, fig_dirpath='data/')

    # generations = train_data[2000:3000]#avb.generate(n_samples=100, batch_size=100)
    # plot_sampled_data(generations, sample_side_size=28, fig_dirpath='data/')
