from keras.layers import Dense, Conv2DTranspose, Conv2D, Concatenate, Dot, Reshape, LocallyConnected1D
from math import sqrt


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
    deflated = Conv2D(filters=n_filters_init, kernel_size=(5, 5), strides=(1, 1),
                      padding='same', activation='relu', name=name_prefix + '_conv_0')(inputs)
    for i in xrange(1, n_deflation_layers):
        deflated = Conv2D(filters=n_filters_init * (2**i), kernel_size=(5, 5), strides=(1, 1),
                          padding='same', activation='relu', name=name_prefix + '_conv_{}'.format(i))(deflated)
    return deflated


def residual_connection(inputs, output_shape, n_hidden):
    raise NotImplementedError


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

    reshaped_noise = Reshape((-1, 1))(noise_input)
    noise_basis_vectors = LocallyConnected1D(filters=16, kernel_size=1,
                                             strides=1, activation='relu', name='enc_noise_f_0')(reshaped_noise)
    noise_basis_vectors = LocallyConnected1D(filters=32, kernel_size=1,
                                             strides=1, activation='relu', name='enc_noise_f_1')(noise_basis_vectors)
    noise_basis_vectors = LocallyConnected1D(filters=latent_dim, kernel_size=1,
                                             strides=1, activation='relu', name='enc_noise_f_2')(noise_basis_vectors)

    convnet_input = Reshape((int(sqrt(data_dim)), int(sqrt(data_dim))))(data_input)
    coefficients = deflating_convolution(convnet_input, n_deflation_layers=3, name_prefix='enc_body')
    coefficients = Reshape((-1,))(coefficients)
    coefficients = Dense(latent_dim)(coefficients)

    linear_combination = Dot(axes=-1)([noise_basis_vectors, coefficients])
    return linear_combination


def mnist_decoder(inputs):
    # use transposed convolutions to inflate the latent space to (?, 32, 32, 64)
    decoder_body = inflating_convolution(inputs, 4, projection_space_shape=(2, 2, 512), name_prefix='dec_body')
    # use single non-padded convolution to shrink the size to (?, 28, 28, 1)
    decoder_body = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                          padding='valid', name='dec_body_conv')(decoder_body)
    # reshape to flatten out the output
    decoder_body = Reshape((-1,), name='dec_body_reshape_out')(decoder_body)
    return decoder_body


def mnist_discriminator(inputs):
    data_input, noise_input = inputs
    discriminator_input = Concatenate(axis=1, name='disc_data_noise_concat')([data_input, noise_input])
    discriminator_body = repeat_dense(discriminator_input, n_layers=4, n_units=1024, name_prefix='disc_body')
    return discriminator_body
