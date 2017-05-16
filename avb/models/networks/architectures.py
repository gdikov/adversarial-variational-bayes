from keras.layers import Dense, Conv2DTranspose


def basic_network(inputs, output_shape, n_hidden=2):
    if not isinstance(output_shape, int):
        raise TypeError("Output shape for the basic network should be a single integer "
                        "for the number of hidden units without activation in the last layer.")
    h = Dense(512, activation='relu')(inputs)
    for i in xrange(n_hidden - 1):
        h = Dense(512, activation='relu')(h)
    h = Dense(output_shape)(h)
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