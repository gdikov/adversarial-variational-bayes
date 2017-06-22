import keras.backend as ker
from keras.layers import Dense, Conv2DTranspose, Conv2D, Dot, Reshape, Multiply, Concatenate, Add, Lambda
from keras.models import Model, Input


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


def inflating_convolution(inputs, n_inflation_layers, projection_space_shape=(4, 4, 32), name_prefix=None):
    assert len(projection_space_shape) == 3, \
        "Projection space shape is {} but should be 3.".format(len(projection_space_shape))
    flattened_space_dim = reduce(lambda x, y: x*y, projection_space_shape)
    projection = Dense(flattened_space_dim, activation=None, name=name_prefix + '_projection')(inputs)
    reshape = Reshape(projection_space_shape, name=name_prefix + '_reshape')(projection)
    depth = projection_space_shape[2]
    inflated = Conv2DTranspose(filters=min(32, depth // 2), kernel_size=(5, 5), strides=(2, 2), activation='relu',
                               padding='same', name=name_prefix + '_transposed_conv_0')(reshape)
    for i in xrange(1, n_inflation_layers):
        inflated = Conv2DTranspose(filters=max(1, depth // 2**(i+1)), kernel_size=(5, 5),
                                   strides=(2, 2), activation='relu', padding='same',
                                   name=name_prefix + '_transpose_conv_{}'.format(i))(inflated)
    return inflated


def deflating_convolution(inputs, n_deflation_layers, n_filters_init=32, noise=None, name_prefix=None):
    def add_linear_noise(x, eps, ind):
        flattened_deflated = Reshape((-1,), name=name_prefix + '_conv_flatten_{}'.format(ind))(x)
        deflated_shape = ker.int_shape(x)
        deflated_size = deflated_shape[1] * deflated_shape[2] * deflated_shape[3]
        noise_transformed = Dense(deflated_size, activation=None,
                                  name=name_prefix + '_conv_noise_dense_{}'.format(ind))(eps)
        added_noise = Add(name=name_prefix + '_conv_add_noise_{}'.format(ind))([noise_transformed, flattened_deflated])
        x = Reshape((deflated_shape[1], deflated_shape[2], deflated_shape[3]),
                    name=name_prefix + '_conv_backreshape_{}'.format(ind))(added_noise)
        return x

    deflated = Conv2D(filters=n_filters_init, kernel_size=(5, 5), strides=(2, 2),
                      padding='same', activation='relu', name=name_prefix + '_conv_0')(inputs)
    if noise is not None:
        deflated = add_linear_noise(deflated, noise, 0)
    for i in xrange(1, n_deflation_layers):
        deflated = Conv2D(filters=n_filters_init * (2**i), kernel_size=(5, 5), strides=(2, 2),
                          padding='same', activation='relu', name=name_prefix + '_conv_{}'.format(i))(deflated)
        # if noise is not None:
        #     deflated = add_linear_noise(deflated, noise, i)
    return deflated


""" Architectures for reproducing paper experiments on the synthetic 4-points dataset. """


def synthetic_encoder(data_dim, noise_dim, latent_dim=2):
    data_input = Input(shape=(data_dim,), name='enc_internal_data_input')
    noise_input = Input(shape=(noise_dim,), name='enc_internal_noise_input')
    data_noise_concat = Concatenate(axis=1, name='enc_data_noise_concatenation')([data_input, noise_input])
    encoder_body = repeat_dense(data_noise_concat, n_layers=2, n_units=256, name_prefix='enc_body')
    latent_factors = Dense(latent_dim, activation=None, name='enc_latent')(encoder_body)
    latent_model = Model(inputs=[data_input, noise_input], outputs=latent_factors, name='enc_internal_model')
    return latent_model


def synthetic_reparametrized_encoder(inputs, latent_dim):
    encoder_body = repeat_dense(inputs, n_layers=2, n_units=256, name_prefix='rep_enc_body')
    latent_mean = Dense(latent_dim, activation=None, name='rep_enc_mean')(encoder_body)
    # since the variance must be positive and this is not easy to restrict, interpret it in the log domain
    latent_log_var = Dense(latent_dim, activation=None, name='rep_enc_var')(encoder_body)
    return latent_mean, latent_log_var


def synthetic_moment_estimation_encoder(data_dim, noise_dim, noise_basis_dim, latent_dim=2):
    noise_input = Input(shape=(noise_basis_dim, noise_dim,), name='enc_internal_noise_input')

    # compute the noise basis vectors by attaching small independent fully connected networks to each noise input from
    # noise_basis ones in total
    def get_inp_row(inputs, **kwargs):
        row = kwargs.get('i', 0)
        # first axis is the batch size, so skip it
        return inputs[:, row]

    noise_basis_vectors = []
    for i in range(noise_basis_dim):
        l = Lambda(get_inp_row, arguments={'i': i}, name='enc_noise_basis_vec_select_{}'.format(i))(noise_input)
        fc = Dense(32, activation='relu', name='enc_noise_basis_body_0_basis_{}'.format(i))(l)
        fc = Dense(32, activation='relu', name='enc_noise_basis_body_1_basis_{}'.format(i))(fc)
        fc = Dense(latent_dim, activation=None, name='enc_noise_basis_body_2_basis_{}'.format(i))(fc)
        noise_basis_vectors.append(fc)

    noise_basis_vectors_model = Model(inputs=noise_input, outputs=noise_basis_vectors,
                                      name='enc_noise_basis_vector_model')

    data_input = Input(shape=(data_dim,), name='enc_internal_data_input')
    # center the input around 0
    centered_data = Lambda(lambda x: 2 * x - 1, name='enc_centering_data_input')(data_input)
    # compute the data embedding using deep convolutional neural network and reshape the output to the noise dim.
    extracted_features = repeat_dense(centered_data, n_layers=2, n_units=256, name_prefix='enc_body')
    latent_0 = Dense(latent_dim, name='enc_coefficients')(extracted_features)
    coefficients = []
    for i in xrange(noise_basis_dim):
        coefficients.append(Dense(latent_dim, activation='elu',
                                  name='enc_coefficients_{}'.format(i))(extracted_features))
    coefficients.append(latent_0)
    coefficients_model = Model(inputs=data_input, outputs=coefficients, name='enc_coefficients_model')

    return coefficients_model, noise_basis_vectors_model


def synthetic_decoder(inputs):
    decoder_body = repeat_dense(inputs, n_layers=2, n_units=256, name_prefix='dec_body')
    return decoder_body


def synthetic_discriminator(data_dim, latent_dim):
    data_input = Input(shape=(data_dim,), name='disc_internal_data_input')
    # center the data around 0 in [-1, 1] as it is in [0, 1].
    centered_data = Lambda(lambda x: 2 * x - 1, name='disc_centering_data_input')(data_input)
    discriminator_body_data = repeat_dense(centered_data, n_layers=2, n_units=256, name_prefix='disc_body_data')

    latent_input = Input(shape=(latent_dim,), name='disc_internal_latent_input')
    discriminator_body_latent = repeat_dense(latent_input, n_layers=2, n_units=256, name_prefix='disc_body_latent')

    discriminator_output = Dot(axes=1, name='disc_output_dot')([discriminator_body_data, discriminator_body_latent])
    discriminator_model = Model(inputs=[data_input, latent_input], outputs=discriminator_output,
                                name='disc_internal_model')
    return discriminator_model


def synthetic_adaptive_prior_discriminator(data_dim, latent_dim):
    data_input = Input(shape=(data_dim,), name='disc_internal_data_input')
    # center the data around 0 in [-1, 1] as it is in [0, 1].
    centered_data = Lambda(lambda x: 2 * x - 1, name='disc_centering_data_input')(data_input)
    discriminator_body_data = repeat_dense(centered_data, n_layers=2, n_units=256, name_prefix='disc_body_data')
    theta = Dense(4*256, activation='relu', name='disc_theta')(discriminator_body_data)
    discriminator_body_data_t = repeat_dense(centered_data, n_layers=2, n_units=256, name_prefix='disc_body_data_t')
    discriminator_body_data_t = Dense(1, activation=None, name='disc_data_squash')(discriminator_body_data_t)

    latent_input = Input(shape=(latent_dim,), name='disc_internal_latent_input')
    discriminator_body_latent = repeat_dense(latent_input, n_layers=2, n_units=256, name_prefix='disc_body_latent')
    sigma = Dense(4*256, activation='relu', name='disc_sigma')(discriminator_body_latent)
    discriminator_body_latent_t = repeat_dense(latent_input, n_layers=2, n_units=256, name_prefix='disc_body_latent_t')
    discriminator_body_latent_t = Dense(1, activation=None, name='disc_latent_squash')(discriminator_body_latent_t)

    merged_data_latent = Multiply(name='disc_mul_sigma_theta')([theta, sigma])
    merged_data_latent = Lambda(lambda x: ker.sum(x, axis=-1), name='disc_add_activ_sig_the')(merged_data_latent)
    discriminator_output = Add(name='disc_add_data_latent_t')([discriminator_body_data_t,
                                                               discriminator_body_latent_t,
                                                               merged_data_latent])
    collapsed_noise = Lambda(lambda x: 0.5 * ker.sum(x ** 2, axis=-1), name='disc_noise_addition')(latent_input)
    discriminator_output = Add(name='disc_add_all_toghether')([discriminator_output, collapsed_noise])
    discriminator_model = Model(inputs=[data_input, latent_input], outputs=discriminator_output,
                                name='disc_internal_model')
    return discriminator_model


""" Architectures for reproducing paper experiments on the MNIST dataset. """


def mnist_encoder(data_dim, noise_dim, latent_dim=8):
    data_input = Input(shape=(data_dim,), name='enc_internal_data_input')
    noise_input = Input(shape=(noise_dim,), name='enc_internal_noise_input')
    # center the input around 0
    centered_data = Lambda(lambda x: 2 * x - 1, name='enc_centering_data_input')(data_input)
    convnet_input = Reshape((28, 28, 1), name='enc_data_reshape')(centered_data)
    conv_output = deflating_convolution(convnet_input, n_deflation_layers=3,
                                        n_filters_init=64, noise=noise_input,
                                        name_prefix='enc_data_body')
    conv_output = Reshape((-1,), name='enc_data_features_reshape')(conv_output)
    conv_output = Dense(800, activation='relu', name='enc_dense_before_latent')(conv_output)
    conv_output = Dense(latent_dim, name='enc_latent_features')(conv_output)
    latent_factors = Model(inputs=[data_input, noise_input], outputs=conv_output, name='enc_internal_model')

    return latent_factors


def mnist_encoder_simple(data_dim, noise_dim, latent_dim=8):
    data_input = Input(shape=(data_dim,), name='enc_internal_data_input')
    noise_input = Input(shape=(noise_dim,), name='enc_internal_noise_input')
    # center the input around 0
    # centered_data = Lambda(lambda x: 2 * x - 1, name='enc_centering_data_input')(data_input)
    # concat_input = Concatenate(axis=-1, name='enc_noise_data_concat')([centered_data, noise_input])
    enc_body = repeat_dense(data_input, n_layers=2, n_units=256, activation='relu', name_prefix='enc_body')
    enc_output = Dense(100, activation='relu', name='enc_dense_before_latent')(enc_body)
    enc_output = Dense(latent_dim, name='enc_latent_features')(enc_output)
    noise_resized = Dense(latent_dim, activation=None, name='enc_noise_resizing')(noise_input)
    enc_output = Add(name='enc_add_noise_data')([enc_output, noise_resized])
    latent_factors = Model(inputs=[data_input, noise_input], outputs=enc_output, name='enc_internal_model')

    return latent_factors


def mnist_reparametrized_encoder(inputs, latent_dim):
    # center the input around 0
    centered_data = Lambda(lambda x: 2 * x - 1, name='enc_centering_data_input')(inputs)
    convnet_input = Reshape((28, 28, 1), name='rep_enc_data_reshape')(centered_data)
    conv_output = deflating_convolution(convnet_input, n_deflation_layers=3,
                                        n_filters_init=64, name_prefix='rep_enc_data_body')
    latent_factors = Reshape((-1,), name='rep_enc_data_features_reshape')(conv_output)
    latent_factors = Dense(800, activation='relu', name='rep_enc_expanding_before_latent')(latent_factors)
    latent_mean = Dense(latent_dim, activation=None, name='rep_enc_mean')(latent_factors)
    # since the variance must be positive and this is not easy to restrict, interpret it in the log domain
    latent_log_var = Dense(latent_dim, activation=None, name='rep_enc_var')(latent_factors)
    return latent_mean, latent_log_var


def mnist_reparametrized_encoder_simple(inputs, latent_dim):
    # center the input around 0
    centered_data = Lambda(lambda x: 2 * x - 1, name='enc_centering_data_input')(inputs)
    enc_body = repeat_dense(centered_data, n_layers=2, n_units=256, activation='relu', name_prefix='enc_body')
    enc_output = Dense(100, activation='relu', name='enc_dense_before_latent')(enc_body)
    latent_mean = Dense(latent_dim, activation=None, name='rep_enc_mean')(enc_output)
    # since the variance must be positive and this is not easy to restrict, interpret it in the log domain
    latent_log_var = Dense(latent_dim, activation=None, name='rep_enc_var')(enc_output)
    return latent_mean, latent_log_var


def mnist_moment_estimation_encoder(data_dim, noise_dim, noise_basis_dim, latent_dim=8):
    noise_input = Input(shape=(noise_basis_dim, noise_dim,), name='enc_internal_noise_input')

    # compute the noise basis vectors by attaching small independent fully connected networks to each noise input from
    # noise_basis ones in total
    def get_inp_row(inputs, **kwargs):
        row = kwargs.get('i', 0)
        # first axis is the batch size, so skip it
        return inputs[:, row]

    noise_basis_vectors = []
    for i in range(noise_basis_dim):
        l = Lambda(get_inp_row, arguments={'i': i}, name='enc_noise_basis_vec_select_{}'.format(i))(noise_input)
        fc = Dense(128, activation='relu', name='enc_noise_basis_body_0_basis_{}'.format(i))(l)
        fc = Dense(128, activation='relu', name='enc_noise_basis_body_1_basis_{}'.format(i))(fc)
        fc = Dense(128, activation='relu', name='enc_noise_basis_body_2_basis_{}'.format(i))(fc)
        fc = Dense(latent_dim, activation=None, name='enc_noise_basis_body_3_basis_{}'.format(i))(fc)
        noise_basis_vectors.append(fc)

    noise_basis_vectors_model = Model(inputs=noise_input, outputs=noise_basis_vectors,
                                      name='enc_noise_basis_vector_model')

    data_input = Input(shape=(data_dim,), name='enc_internal_data_input')
    assert data_dim == 28 ** 2, "MNIST data should be flattened to 784-dimensional vectors."
    # center the input around 0
    centered_data = Lambda(lambda x: 2 * x - 1, name='enc_centering_data_input')(data_input)
    # compute the data embedding using deep convolutional neural network and reshape the output to the noise dim.
    convnet_input = Reshape((28, 28, 1), name='enc_data_reshape')(centered_data)
    # add noise to the convolutions
    # partial_noise = Reshape((-1,), name='enc_noise_addition_conv')(noise_input)
    coefficients = deflating_convolution(convnet_input, n_deflation_layers=3,
                                         n_filters_init=64, name_prefix='enc_data_body')
    coefficients = Reshape((-1,), name='enc_data_features_reshape')(coefficients)
    extracted_features = Dense(800, activation='relu', name='enc_expanding_before_latent')(coefficients)

    latent_0 = Dense(latent_dim, name='enc_coefficients')(extracted_features)
    coefficients = []
    for i in xrange(noise_basis_dim):
        coefficients.append(Dense(latent_dim, activation='elu',
                                  name='enc_coefficients_{}'.format(i))(extracted_features))
    coefficients.append(latent_0)
    coefficients_model = Model(inputs=data_input, outputs=coefficients, name='enc_coefficients_model')

    return coefficients_model, noise_basis_vectors_model


def mnist_decoder(inputs):
    decoder_body = Dense(200, activation='relu', name='dec_body_initial_dense')(inputs)
    # use transposed convolutions to inflate the latent space to (?, 32, 32, 8)
    decoder_body = inflating_convolution(decoder_body, 3, projection_space_shape=(4, 4, 32), name_prefix='dec_body')
    # use single non-padded convolution to shrink the size to (?, 28, 28, 1)
    decoder_body = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                          padding='valid', name='dec_body_conv')(decoder_body)
    # reshape to flatten out the output
    decoder_body = Reshape((-1,), name='dec_body_reshape_out')(decoder_body)
    return decoder_body


def mnist_decoder_simple(inputs):
    decoder_body = repeat_dense(inputs, n_layers=3, n_units=512, name_prefix='dec_body')
    return decoder_body


def mnist_adaptive_prior_discriminator(data_dim, latent_dim):
    data_input = Input(shape=(data_dim,), name='disc_internal_data_input')
    # center the data around 0 in [-1, 1] as it is in [0, 1].
    centered_data = Lambda(lambda x: 2 * x - 1, name='disc_centering_data_input')(data_input)
    discriminator_body_data = repeat_dense(centered_data, n_layers=3, n_units=512, name_prefix='disc_body_data')
    theta = Dense(4*512, activation='relu', name='disc_theta')(discriminator_body_data)
    discriminator_body_data_t = repeat_dense(centered_data, n_layers=3, n_units=512, name_prefix='disc_body_data_t')
    discriminator_body_data_t = Dense(1, activation=None, name='disc_data_squash')(discriminator_body_data_t)

    latent_input = Input(shape=(latent_dim,), name='disc_internal_latent_input')
    discriminator_body_latent = repeat_dense(latent_input, n_layers=3, n_units=512, name_prefix='disc_body_latent')
    sigma = Dense(4*512, activation='relu', name='disc_sigma')(discriminator_body_latent)
    discriminator_body_latent_t = repeat_dense(latent_input, n_layers=3, n_units=512, name_prefix='disc_body_latent_t')
    discriminator_body_latent_t = Dense(1, activation=None, name='disc_latent_squash')(discriminator_body_latent_t)

    merged_data_latent = Multiply(name='disc_mul_sigma_theta')([theta, sigma])
    merged_data_latent = Lambda(lambda x: ker.sum(x, axis=-1), name='disc_add_activ_sig_the')(merged_data_latent)
    discriminator_output = Add(name='disc_add_data_latent_t')([discriminator_body_data_t,
                                                               discriminator_body_latent_t,
                                                               merged_data_latent])
    collapsed_noise = Lambda(lambda x:  0.5 * ker.sum(x**2, axis=-1), name='disc_noise_addition')(latent_input)
    discriminator_output = Add(name='disc_add_all_toghether')([discriminator_output, collapsed_noise])
    discriminator_model = Model(inputs=[data_input, latent_input], outputs=discriminator_output,
                                name='disc_internal_model')
    return discriminator_model


def mnist_discriminator_simple(data_dim, latent_dim):
    data_input = Input(shape=(data_dim,), name='disc_internal_data_input')
    # center the data around 0 in [-1, 1] as it is in [0, 1].
    centered_data = Lambda(lambda x: 2 * x - 1, name='disc_centering_data_input')(data_input)
    discriminator_body_data = repeat_dense(centered_data, n_layers=3, n_units=512, name_prefix='disc_body_data')

    latent_input = Input(shape=(latent_dim,), name='disc_internal_latent_input')
    discriminator_body_latent = repeat_dense(latent_input, n_layers=4, n_units=512, name_prefix='disc_body_latent')

    discriminator_output = Dot(axes=-1, name='disc_dot_sigma_theta')([discriminator_body_data,
                                                                      discriminator_body_latent])

    discriminator_model = Model(inputs=[data_input, latent_input], outputs=discriminator_output,
                                name='disc_internal_model')
    return discriminator_model


get_network_by_name = {'encoder': {'synthetic': synthetic_encoder,
                                   'mnist': mnist_encoder,
                                   'mnist_simple': mnist_encoder_simple},
                       'reparametrised_encoder': {'synthetic': synthetic_reparametrized_encoder,
                                                  'mnist': mnist_reparametrized_encoder,
                                                  'mnist_simple': mnist_reparametrized_encoder_simple},
                       'moment_estimation_encoder': {'synthetic': synthetic_moment_estimation_encoder,
                                                     'mnist': mnist_moment_estimation_encoder},
                       'decoder': {'synthetic': synthetic_decoder,
                                   'mnist': mnist_decoder,
                                   'mnist_simple': mnist_decoder_simple},
                       'discriminator': {'synthetic': synthetic_discriminator,
                                         'mnist': mnist_discriminator_simple,
                                         'mnist_simple': mnist_discriminator_simple},
                       'adaptive_prior_discriminator': {'synthetic': synthetic_adaptive_prior_discriminator,
                                                        'mnist': mnist_adaptive_prior_discriminator,
                                                        'mnist_simple': mnist_discriminator_simple}
                       }
