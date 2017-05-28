from keras.layers import Dense, Dot, Activation, Concatenate
from keras.models import Model, Input
from architectures import repeat_dense


class Discriminator(object):
    """
    Discriminator model is adversarially trained against the encoder in order to account 
    for a D_KL(q(z|x) || p(z)) term in the variational loss (see AVB paper, page 3). The discriminator
    architecture takes as input samples from the joint probability distribution of the data `x` and a approximate
    posterior `z` and from the joint of the data and the prior over `z`:
     
       ----> -----------
       |     | Encoder |
       |     -----------
       |         |
       |    Approx. posterior --> | 
       |                          |---> (x, z') --|
       -------------------------> |               |
       |                                          |     -----------
      Data                                        | --> | Decoder | --> T(x,z) regression output
       |                                          |     -----------
       -------------------------> |               |
                                  |---> (x, z)  --|
       Prior p(z): N(0,I) ------> |
       
    """
    def __init__(self, data_dim, latent_dim):
        """
        Args:
            data_dim: int, the flattened dimensionality of the data space
            latent_dim: int, the flattened dimensionality of the latent space
        """
        discriminator_input_data = Input(shape=(data_dim,), name='disc_input_data')
        discriminator_input_latent = Input(shape=(latent_dim,), name='disc_input_latent')

        discriminator_body_data = repeat_dense(discriminator_input_data, num_layers=2, num_units=512,
                                               name_prefix='disc_body_data')

        discriminator_body_latent = repeat_dense(discriminator_input_latent, num_layers=2, num_units=512,
                                                 name_prefix='disc_body_latent')

        merged_data_latent = Dot(axes=1, name='disc_merge')([discriminator_body_data, discriminator_body_latent])
        discriminator_output = Activation(activation='sigmoid', name='disc_output')(merged_data_latent)

        # merged_data_latent = Concatenate(axis=1, name='disc_merge')([discriminator_input_data, discriminator_input_latent])
        #
        # discriminator_body = repeat_dense(merged_data_latent, num_layers=2, num_units=256, name_prefix='disc_body')
        #
        # discriminator_output = Dense(1, activation='sigmoid', name='disc_out')(discriminator_body)

        self.discriminator_model = Model(inputs=[discriminator_input_data, discriminator_input_latent],
                                         outputs=discriminator_output,
                                         name='discriminator')

    def __call__(self, *args, **kwargs):
        """
        Make the Discriminator model callable on a list of Inputs (coming from the AVB model)
        
        Args:
            *args: a list of Input layers
            **kwargs: 

        Returns:
            A trainable Discriminator model. 
        """
        return self.discriminator_model(args[0])
