from tensorflow.contrib.distributions import Bernoulli
from keras.layers import Lambda, Dense
from keras.models import Model, Input


class Decoder(object):
    def __init__(self, latent_dim, output_dim):
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        real_data = Input(shape=(self.output_dim,), name='decoder_ll_estimator_data_input')
        latent_encoding = Input(shape=(self.latent_dim,), name='decoder_latent_input')

        self.generator = self._build_generator_model()(latent_encoding)

        log_probs = Lambda(lambda x: Bernoulli(probs=self.generator, validate_args=False).log_prob(x))(real_data)

        self.ll_estimator = Model(inputs=[real_data, latent_encoding], outputs=log_probs, name='LL-Decoder')

    def _build_generator_model(self):

        latent_encoding = Input(shape=(self.latent_dim,), name='generator_latent_input')

        generator_body = Dense(256, activation='relu')(latent_encoding)
        generator_body = Dense(256, activation='relu')(generator_body)

        sampler_params = Dense(self.output_dim, activation='sigmoid', name='decoder_sampler_params')(generator_body)
        sampler_params = Lambda(lambda x: 1e-6 + (1 - 2*1e-6) * x)(sampler_params)

        generator_model = Model(inputs=latent_encoding, outputs=sampler_params, name='Sampling-Decoder')
        return generator_model

    def __call__(self, *args, **kwargs):
        is_learninig = kwargs.get('is_learning', True)
        if is_learninig:
            return self.ll_estimator(args[0])
        else:
            return self.generator(args[0])
