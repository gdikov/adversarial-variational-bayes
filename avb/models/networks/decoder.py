from tensorflow.contrib.distributions import Bernoulli
from keras.layers import Lambda, Dense
from keras.models import Model, Input


class Decoder(object):
    def __init__(self, latent_dim, data_dim):
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        real_data = Input(shape=(self.data_dim,), name='dec_ll_estimator_data_input')
        latent_encoding = Input(shape=(self.latent_dim,), name='dec_latent_input')

        generator_body = Dense(512, activation='relu', name='dec_body1')(latent_encoding)
        generator_body = Dense(512, activation='relu', name='dec_body2')(generator_body)

        sampler_params = Dense(self.data_dim, activation='sigmoid', name='dec_sampler_params')(generator_body)
        sampler_params = Lambda(lambda x: 1e-6 + (1 - 2e-6) * x, name='dec_probs_clipper')(sampler_params)

        log_probs = Lambda(lambda x: Bernoulli(probs=x[0], name='dec_bernoulli').log_prob(x[1]),
                           name='dec_bernoulli_logprob')([sampler_params, real_data])

        self.generator = Model(inputs=latent_encoding, outputs=sampler_params, name='dec_sampling')
        self.ll_estimator = Model(inputs=[real_data, latent_encoding], outputs=log_probs, name='dec_trainable')

    def __call__(self, *args, **kwargs):
        is_learninig = kwargs.get('is_learning', True)
        if is_learninig:
            return self.ll_estimator(args[0])
        else:
            return self.generator(args[0])
