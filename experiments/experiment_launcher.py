from utils.data_utils.datasets import load_npoints
from avb.models.avb import AdversarialVariationalBayes
from visualization.latent_space import plot_latent_2d

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
    data = load_npoints(n=4)
    train_data, train_labels = data['data'], data['target']
    avb = AdversarialVariationalBayes(data_dim=4, latent_dim=2, noise_dim=4)
    avb.fit(train_data, batch_size=512, epochs=20)
    latent_vars = avb.infer(train_data, batch_size=train_data.shape[0])
    plot_latent_2d(latent_vars, train_labels)
    generations = avb.generate(n_points=10)
    print(generations)
