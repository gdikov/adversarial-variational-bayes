from third_party.ite.cost import MDKL_HSCE as D_KL
import numpy as np


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


def d_kl_from_samples(posterior_samples, data_samples):
    kl_div = KLDivergenceEstimator()
    kl_div.estimate(posterior_samples, data_samples)


def d_kl_against_normal_prior(posterior_samples):
    # to estimate the marginal posterior q(z) = integral q(z|x)p(x) dx = integral q(z,x) dx
    kl_div = KLDivergenceEstimator(reference_dist='standard_normal')
    kl_div.estimate(posterior_samples)


def reconstruction_error(true_data, reconstructed_data, ):
    # measure binary cross-entropy in the case of a Bernoulli decoder and binarized input data.
    x_entropy = np.mean(np.sum(true_data * np.log(reconstructed_data) +
                               (1 - true_data) * np.log(1 - reconstructed_data), axis=-1))
