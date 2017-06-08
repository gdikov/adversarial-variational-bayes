import keras.backend as ker

from keras.layers import Dense, Conv2DTranspose, Conv2D, Concatenate, Dot, Reshape, LocallyConnected1D, Lambda


def repeat_dense(inputs, n_layers, n_units=256, activation='relu', name_prefix=None):
    """
    Repeat Dense Keras layers, attached to given input. 
    
    Args:
        inputs: A Keras layer or Tensor preceding the repeated layers 
        n_layers: number of layers to repeat
        n_units: number of units in each layer 
        activation: the activation in each layer
        name_prefix: the prefix of the named layers. A `_i` will be be appended automatically, 
            where i is the layer number, starting from 0.

    Returns:
        The last appended Keras layer. 
    """
    if n_units < 1 or n_layers < 1:
        raise ValueError('`n_layers` and `n_units` must be >= 1, '
                         'found {} and {} respectively'.format(n_layers, n_units))
    n_units = int(n_units)
    n_layers = int(n_layers)
    name_prefix = name_prefix or 'rep_{}_dim{}'.format(activation, n_units)
    h = Dense(n_units, activation=activation, name=name_prefix + '_0')(inputs)
    for i in xrange(1, n_layers):
        h = Dense(n_units, activation=activation, name=name_prefix + '_{}'.format(i))(h)
    return h


def inflating_convolution(inputs, n_inflation_layers, projection_space_shape=(4, 4, 1024), name_prefix=None):
    assert len(projection_space_shape) == 3, \
        "Projection space shape is {} but should be 3.".format(len(projection_space_shape))
    flattened_space_dim = reduce(lambda x, y: x*y, projection_space_shape)
    projection = Dense(flattened_space_dim, activation=None, name=name_prefix + '_projection')(inputs)
    reshape = Reshape(projection_space_shape, name=name_prefix + '_reshape')(projection)
    depth = projection_space_shape[2]
    inflated = Conv2DTranspose(filters=depth // 2, kernel_size=(5, 5), strides=(2, 2), activation='relu',
                               padding='same', name=name_prefix + '_transposed_conv_0')(reshape)
    for i in xrange(1, n_inflation_layers):
        inflated = Conv2DTranspose(filters=depth // max(1, 2**(i+1)), kernel_size=(5, 5),
                                   strides=(2, 2), activation='relu', padding='same',
                                   name=name_prefix + '_transpose_conv_{}'.format(i))(inflated)
    return inflated


def deflating_convolution(inputs, n_deflation_layers, n_filters_init=32, name_prefix=None):
    deflated = Conv2D(filters=n_filters_init, kernel_size=(5, 5), strides=(2, 2),
                      padding='same', activation='relu', name=name_prefix + '_conv_0')(inputs)
    for i in xrange(1, n_deflation_layers):
        deflated = Conv2D(filters=n_filters_init * (2**i), kernel_size=(5, 5), strides=(2, 2),
                          padding='same', activation='relu', name=name_prefix + '_conv_{}'.format(i))(deflated)
    return deflated


""" Architectures for reproducing paper experiments on the synthetic 4-points dataset. """


def synthetic_encoder(inputs, latent_dim):
    data_input, noise_input = inputs
    encoder_input = Concatenate(axis=1, name='enc_data_noise_concat')([data_input, noise_input])
    encoder_body = repeat_dense(encoder_input, n_layers=2, n_units=256, name_prefix='enc_body')
    latent_factors = Dense(latent_dim, activation=None, name='enc_latent')(encoder_body)
    return latent_factors


def synthetic_reparametrized_encoder(inputs, latent_dim):
    encoder_body = repeat_dense(inputs, n_layers=2, n_units=256, name_prefix='rep_enc_body')
    latent_mean = Dense(latent_dim, activation=None, name='rep_enc_mean')(encoder_body)
    # since the variance must be positive and this is not easy to restrict, interpret it in the log domain
    latent_log_var = Dense(latent_dim, activation=None, name='rep_enc_var')(encoder_body)
    return latent_mean, latent_log_var


def synthetic_decoder(inputs):
    decoder_body = repeat_dense(inputs, n_layers=2, n_units=256, name_prefix='dec_body')
    return decoder_body


def synthetic_discriminator(inputs):
    data_input, latent_input = inputs
    discriminator_body_data = repeat_dense(data_input, n_layers=2, n_units=256, name_prefix='disc_body_data')
    discriminator_body_latent = repeat_dense(latent_input, n_layers=2, n_units=256, name_prefix='disc_body_latent')
    merged_data_latent = Dot(axes=1, name='disc_merge')([discriminator_body_data, discriminator_body_latent])
    return merged_data_latent


""" Architectures for reproducing paper experiments on the MNIST dataset. """


def mnist_encoder(inputs, latent_dim=8):
    data_input, noise_input = inputs
    data_dim = data_input.shape[1]
    assert data_dim == 28 ** 2, "MNIST data should be flattened to a 784-dimensional vectors."

    convnet_input = Reshape((28, 28, 1), name='enc_reshape_data')(data_input)
    latent_factors = deflating_convolution(convnet_input, n_deflation_layers=3, name_prefix='enc_body')
    latent_factors = Reshape((-1,), name='enc_reshape_latent')(latent_factors)
    latent_factors = Dense(latent_dim, name='enc_latent')(latent_factors)

    return latent_factors


def mnist_moment_estimation_encoder(inputs, latent_dim=8):
    data_input, noise_input = inputs
    data_dim = data_input.shape[1].value
    noise_dim = noise_input.shape[1].value

    assert data_dim == 28**2, "MNIST data should be flattened to a 784-dimensional vectors."
    # compute the noise basis vectors by attaching small independent fully connected networks to each noise scalar input
    reshaped_noise = Reshape((-1, 1), name='enc_noise_reshape')(noise_input)
    noise_basis_vectors = LocallyConnected1D(filters=16, kernel_size=1,
                                             strides=1, activation='relu', name='enc_noise_f_0')(reshaped_noise)
    noise_basis_vectors = LocallyConnected1D(filters=32, kernel_size=1,
                                             strides=1, activation='relu', name='enc_noise_f_1')(noise_basis_vectors)
    noise_basis_vectors = LocallyConnected1D(filters=latent_dim, kernel_size=1,
                                             strides=1, activation='relu', name='enc_noise_f_2')(noise_basis_vectors)
    assert noise_basis_vectors.get_shape().as_list() == [None, noise_dim, latent_dim]

    # compute the data embedding using deep convolutional neural network and reshape the output to the noise dim.
    convnet_input = Reshape((28, 28, 1), name='enc_data_reshape')(data_input)
    coefficients = deflating_convolution(convnet_input, n_deflation_layers=2,
                                         n_filters_init=4, name_prefix='enc_data_body')
    coefficients = Reshape((-1,), name='enc_coefficients_reshape')(coefficients)
    coefficients = Dense(noise_dim, name='enc_coefficients')(coefficients)

    # compute empirical mean as the batchsize-wise mean of all noise vectors
    empirical_mean_basis_vectors = ker.mean(noise_basis_vectors, axis=0)
    # parametrise the posterior moment as described in the AVB paper
    mean = Lambda(lambda x: ker.dot(x[0], x[1]), name='enc_moments_mean')([coefficients, empirical_mean_basis_vectors])

    # do the same for the empirical variance
    empirical_var_basis_vectors = ker.var(noise_basis_vectors, axis=0)
    # similar posterior parametrization for the variance
    var = Lambda(lambda x: ker.dot(x[0]**2, x[1]), name='enc_moments_var')([coefficients, empirical_var_basis_vectors])

    linear_combination = Lambda(lambda x: ker.sum(x[0][:, :, None] * x[1], axis=1),
                                name='enc_basis_vec_coeff_dot')([coefficients, noise_basis_vectors])

    return linear_combination, mean, var


def mnist_decoder(inputs):
    # use transposed convolutions to inflate the latent space to (?, 32, 32, 64)
    decoder_body = inflating_convolution(inputs, 2, projection_space_shape=(2, 2, 128), name_prefix='dec_body')
    # use single non-padded convolution to shrink the size to (?, 28, 28, 1)
    decoder_body = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                          padding='valid', name='dec_body_conv')(decoder_body)
    # reshape to flatten out the output
    decoder_body = Reshape((-1,), name='dec_body_reshape_out')(decoder_body)
    return decoder_body


def mnist_discriminator(inputs):
    data_input, latent_input = inputs
    discriminator_body_data = repeat_dense(data_input, n_layers=2, n_units=256, name_prefix='disc_body_data')
    discriminator_body_latent = repeat_dense(latent_input, n_layers=2, n_units=256, name_prefix='disc_body_latent')
    merged_data_latent = Dot(axes=1, name='disc_merge')([discriminator_body_data, discriminator_body_latent])
    return merged_data_latent


get_network_by_name = {'encoder': {'synthetic': synthetic_encoder,
                                   'mnist': mnist_encoder},
                       'reparametrised_encoder': {'synthetic': synthetic_reparametrized_encoder},
                       'moment_estimation_encoder': {'mnist': mnist_moment_estimation_encoder},
                       'decoder': {'synthetic': synthetic_decoder,
                                   'mnist': mnist_decoder},
                       'discriminator': {'synthetic': synthetic_discriminator,
                                         'mnist': mnist_discriminator}}
