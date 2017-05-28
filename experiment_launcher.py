from avb.model_trainer import AVBModelTrainer
from avb.models.avb import AdversarialVariationalBayes
from utils.datasets import load_npoints, load_mnist
from utils.visualisation import plot_latent_2d, plot_sampled_data
import numpy as np
# Variational Inferece:
#
#   * toy example from Figure 1
#   * Eight Schools posterior approximation
#
#
# Generative:
#
#   * 4-points syntetic
#   * MNIST
#   * CelebA

if __name__ == '__main__':
    # data_dim = 9
    # data = load_npoints(n=data_dim)
    data_dim = 28*28
    data = load_mnist("../data/MNIST/")
    train_data, train_labels = data['data'], data['target']
    avb = AdversarialVariationalBayes(data_dim=data_dim, latent_dim=2, noise_dim=data_dim/28)

    # trainer = AVBModelTrainer(model=avb)
    # trainer.start_training(train_data, batch_size=512, epochs=300)
    avb.fit(train_data, batch_size=10000, epochs=300)
    latent_vars = avb.infer(train_data, batch_size=10000)
    np.save('latent.npy', latent_vars)
    plot_latent_2d(latent_vars, train_labels)

    generations = avb.generate(n_samples=100, batch_size=100)
    np.save('generations.npy', generations)
    plot_sampled_data(generations, sample_side_size=28)
