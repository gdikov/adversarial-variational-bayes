import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_latent_2d(latent_vars, target=None):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(6, 6))
    if target is not None:
        plt.scatter(latent_vars[:, 0], latent_vars[:, 1], c=target)
    else:
        raise NotImplementedError
    plt.colorbar()
    plt.savefig('latent_space.png')
    # plt.show()


def plot_sampled_data(data, sample_side_size):
    data = data.reshape(-1, sample_side_size, sample_side_size)
    data_size = data.shape[0]
    samples_per_fig_side = int(np.sqrt(data_size))
    data = data[:samples_per_fig_side**2].reshape(samples_per_fig_side, samples_per_fig_side,
                                                  sample_side_size, sample_side_size)
    data = np.concatenate(np.concatenate(data, axis=1), axis=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='Greys_r')
    plt.savefig('sampled_data.png')
    # plt.show()
