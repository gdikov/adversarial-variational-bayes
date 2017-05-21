from utils.data_utils.datasets import load_npoints
from avb.models.avb import AdversarialVariationalBayes


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
    avb = AdversarialVariationalBayes(data_dim=4, latent_dim=2)
    avb.fit(data['data'])
