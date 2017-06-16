from numpy import load as load_array
from os.path import join as path_join
from avb.metrics import evidence_lower_bound, normality_of_marginal_posterior, reconstruction_error, data_log_likelihood
from avb.utils.datasets import load_npoints, load_mnist


if __name__ == '__main__':
    model_dir = "output/models/synthetic"
    latent_samples = load_array(path_join(model_dir, 'latent_samples.npy'))
    reconstructed_samples = load_array(path_join(model_dir, 'reconstructed_samples.npy'))
    generated_samples = load_array(path_join(model_dir, 'generated_samples.npy'))
    # data = load_mnist(binarised=True, one_hot=False)
    data = load_npoints(4)
    true_samples, targets = data['data'], data['target']
    elbo = evidence_lower_bound(true_samples=true_samples,
                                reconstructed_samples=reconstructed_samples,
                                latent_samples=latent_samples,
                                targets=targets)
    print("Estimated ELBO = {}".format(elbo))
    kl_marginal_prior = normality_of_marginal_posterior(latent_samples)
    print("KL(q(z) || p(z)) = {}".format(kl_marginal_prior))
    reconstrction_loss = reconstruction_error(true_samples=true_samples,
                                              reconstructed_samples_probs=reconstructed_samples)
    print("Reconstruction error = {}".format(reconstrction_loss))
    data_ll = data_log_likelihood(generated_samples)
    print("Data log likelihood = {}".format(data_ll))
