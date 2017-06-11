from numpy import load as load_array
from os.path import join as path_join
from avb.utils.logger import logger
from avb.metrics import evidence_lower_bound
from avb.utils.datasets import load_npoints


if __name__ == '__main__':
    model_dir = "output/models/synthetic"
    latent_samples = load_array(path_join(model_dir, 'latent_samples.npy'))
    generated_samples = load_array(path_join(model_dir, 'reconstructed_samples.npy'))
    data = load_npoints(4)
    true_samples, targets = data['data'], data['target']
    elbo = evidence_lower_bound(true_samples=true_samples,
                                reconstructed_samples=generated_samples,
                                latent_samples=latent_samples,
                                targets=targets)
    logger.info("Estimated ELBO is: {}".format(elbo))
