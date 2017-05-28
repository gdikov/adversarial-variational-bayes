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
    data_dim = 4
    data = load_npoints(n=data_dim)
    train_data, train_labels = data['data'], data['target']
    avb = AdversarialVariationalBayes(data_dim=data_dim, latent_dim=2, noise_dim=data_dim)
    avb.fit(train_data, batch_size=512, epochs=1000)
    latent_vars = avb.infer(train_data, batch_size=train_data.shape[0])
    plot_latent_2d(latent_vars, train_labels)
    # generations = avb.generate(n_points=100)
    # print(generations)
