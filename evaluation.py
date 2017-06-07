import numpy as np
from avb.metrics import d_kl_against_normal_prior, reconstruction_error
from utils.datasets import load_npoints


if __name__ == '__main__':
    samples = np.load('output/models/4points/sampled_data.npy')
    kl_div = d_kl_against_normal_prior(samples)
    print("Estimated KL divergence between the standard normal prior and the posterior is: {}".format(kl_div))

    real_data = load_npoints(n=4)
    r_err = reconstruction_error(real_data, samples)
    print("Estimated reconstruction error is: {}".format(r_err))
