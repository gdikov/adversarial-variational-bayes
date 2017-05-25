import matplotlib.pyplot as plt


def plot_latent_2d(latent_vars, target=None):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(6, 6))
    if target is not None:
        plt.scatter(latent_vars[:, 0], latent_vars[:, 1], c=target)
    else:
        raise NotImplementedError
    plt.colorbar()
    plt.show()
