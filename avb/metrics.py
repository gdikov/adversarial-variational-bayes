import numpy as np
import logging
logger = logging.getLogger(__name__)
try:
    from third_party.ite.cost import MDKL_HSCE as D_KL
except ImportError:
    D_KL = None
    logger.error("Cannot import ITE package (probably) for using a Python 2 interpreter. "
                 "Using any KL divergence based algorithm will result throw an exception.")


class KLDivergenceEstimator(object):
    def __init__(self, reference_dist='standard_normal'):
        self.estimator = D_KL.estimation
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


def evidence_lower_bound(true_samples, reconstructed_samples, latent_samples):
    reconstruction_ll = reconstruction_log_likelihood(true_samples, reconstructed_samples)
    kl_div = d_kl_against_diag_normal(latent_samples)
    elbo = -kl_div + reconstruction_ll
    return elbo
