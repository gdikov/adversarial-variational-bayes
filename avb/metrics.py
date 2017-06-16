import numpy as np
import logging
import sys
import os

logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.getcwd(), 'third_party'))
try:
    from third_party.ite.cost import MDKL_HSCE as D_KL
except ImportError:
    D_KL = None
    logger.error("Cannot import ITE package, expected to be found under `third_party/ite`")
if sys.version_info < (3,):
    logger.error("KL divergence estimation will fail for it is based on the ITE package which is "
                 "supported only with a Python 3 interpreter. Detected interpreter is {}".format(sys.version_info))


class KLDivergenceEstimator(object):
    def __init__(self, reference_dist='standard_normal'):
        self.estimator = D_KL().estimation
        if reference_dist == 'standard_normal':
            self.ref_distribution = np.random.randn
        else:
            raise NotImplementedError

    def estimate(self, samples_q, samples_p=None, n_samples_p=1000):
        dim = samples_q.shape[1]
        if samples_p is None:
            samples_p = self.ref_distribution(n_samples_p, dim)
        div_estimate = self.estimator(samples_q, samples_p)
        return div_estimate


def kl_divergence(p_samples, q_samples):
    kl_div = KLDivergenceEstimator()
    div = kl_div.estimate(p_samples, q_samples)
    return div


def d_kl_against_diag_normal(samples, normal_mean=np.array([0.]), normal_variance=np.array([1.])):
    normal_samples = np.random.normal(loc=normal_mean, scale=normal_variance, size=samples.shape)
    return kl_divergence(samples, normal_samples)


def reconstruction_log_likelihood(true_samples, estimated_params):
    assert true_samples.shape == estimated_params.shape
    assert true_samples.ndim == 2, "Provide a 2-dim array with a batch size and sample probabilities axes."
    log_probs = true_samples * np.log(estimated_params) + (1 - true_samples) * np.log(1 - estimated_params)
    return np.mean(np.array([log_probs]), axis=1)


def reconstruction_error(true_samples, reconstructed_samples_probs):
    mean_cross_entropy = np.mean(-reconstruction_log_likelihood(true_samples, reconstructed_samples_probs))
    return mean_cross_entropy


def data_log_likelihood(generated_samples):
    dll = np.log(np.mean(np.mean(generated_samples, axis=0)))
    return dll


def evidence_lower_bound(true_samples, reconstructed_samples, latent_samples, targets=None):
    data_size = true_samples.shape[0]
    if targets is None:
        targets = np.zeros(data_size)
    groups = np.unique(targets)
    reconstruction_ll = 0.
    kl_div = 0.
    for ids in [targets == g for g in groups]:
        reconstruction_ll += np.mean(reconstruction_log_likelihood(true_samples[ids], reconstructed_samples[ids]))
        kl_div += d_kl_against_diag_normal(latent_samples[ids])
    elbo = (-kl_div + reconstruction_ll) * (1. * np.sum(ids) / data_size)
    return elbo


def normality_of_marginal_posterior(latent_samples):
    kl_marginal_posterior_prior = d_kl_against_diag_normal(latent_samples)
    return kl_marginal_posterior_prior
