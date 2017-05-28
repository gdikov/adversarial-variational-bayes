from keras.layers import Dense, Conv2DTranspose


def repeat_dense(input, num_layers, num_units=256, activation='relu', name_prefix=None):
    """
    Repeat Dense Keras layers, attached to given input. 
    
    Args:
        input: A Keras layer or Tensor preceding the repeated layers 
        num_layers: number of layers to repeat
        num_units: number of units in each layer 
        activation: the activation in each layer
        name_prefix: the prefix of the named layers. A `_i` will be be appended automatically, 
            where i is the layer number, starting from 0.

    Returns:
        The last appended Keras layer. 
    """
    if num_units < 1 or num_layers < 1:
        raise ValueError('`num_layers` and `num_units` must be >= 1, '
                         'found {} and {} respectively'.format(num_layers, num_units))
    num_units = int(num_units)
    num_layers = int(num_layers)
    name_prefix = name_prefix or 'rep_{}_dim{}'.format(activation, num_units)
    h = Dense(num_units, activation=activation, name=name_prefix + '_0')(input)
    for i in xrange(num_layers - 1):
        h = Dense(num_units, activation=activation, name=name_prefix + '_' + str(i+1))(h)
    return h


def dcgan_decoder(inputs):
    projection = Dense(4*4*1024, activation=None)(inputs)
    inflation = Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2))(projection)    # output is (N, 512, 8, 8)
    inflation = Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2))(inflation)     # output is (N, 256, 16, 16)
    inflation = Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2))(inflation)     # output is (N, 128, 32, 32)
    inflation = Conv2DTranspose(1, kernel_size=(5, 5), strides=(1, 1))(inflation)      # output is (N, 1, 28, 28)
    return inflation


def resnet_encoder(inputs, output_shape, n_hidden):
    pass