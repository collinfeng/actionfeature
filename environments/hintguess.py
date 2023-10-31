import jax
import jax.numpy as jnp
from functools import partial


class HintGuessEnv(object):
    def __init__(self, config: dict):
        self.N = config["N"]
        self.batch_size = config["batch_size"]
        self.feature_dim = config["feature_dim"]
        self.cardmappings = self.idx2hots(self.N, self.feature_dim)

    def idx2hots(self, N, feature_dim):
        idx = jnp.arange(feature_dim**2)
        cardmappings = []
        for i in range(feature_dim**2):
            f1 = jax.nn.one_hot(i//3, feature_dim)
            f2 = jax.nn.one_hot(i%3, feature_dim)
            cardmappings.append(jnp.concatenate((f1, f2), axis=-1))
        return jnp.stack(cardmappings)


    @partial(jax.jit, static_argnums=(0,))
    def get_observation(self, rng):
        '''
        hand dim: batch, hand, feature
        '''
        _, rng, H_rng, tgt_rng = jax.random.split(rng, 4)
        tgt_idx = jax.random.randint(key = tgt_rng, shape=(self.batch_size,), minval=0, maxval=self.N)
        H = jax.random.randint(key=H_rng, shape=(self.batch_size, self.N, 4), minval=0, maxval=self.feature_dim)
        tgt_twohot = jax.nn.one_hot(H[jnp.arange(self.batch_size), tgt_idx, 2:], self.feature_dim).reshape(self.batch_size, self.feature_dim*2)
        H1_twohot = jax.nn.one_hot(H[:, :, :2], self.feature_dim).reshape(self.batch_size, self.N, self.feature_dim*2)
        H2_twohot = jax.nn.one_hot(H[:, :, 2:], self.feature_dim).reshape(self.batch_size, self.N, self.feature_dim*2)
        return (tgt_twohot, H1_twohot, H2_twohot)

    @partial(jax.jit, static_argnums=(0,))
    def init_game(self, rng):
        '''
        The get_observation function with non-repeating hand
        Restricted Handsize < Max possible card choices
        Dim: batch, hand, feature
        '''
        def generate_hands(rng):
            _, rng0, rng1, rng2 = jax.random.split(rng, 4)
            sp_index = jax.random.randint(key=rng0, shape=(1,), minval=0, maxval=self.N)
            h1 = jax.random.permutation(rng1, jnp.arange(self.feature_dim**2))[:self.N]
            h2 = jax.random.permutation(rng2, jnp.arange(self.feature_dim**2))[:self.N]
            h1_twohot = self.cardmappings[h1, :]
            h2_twohot = self.cardmappings[h2, :]
            sp_twohot = h2_twohot[sp_index, :].reshape(-1)
            return h1_twohot, h2_twohot, sp_twohot

        rngs = jax.random.split(rng, self.batch_size)
        v_generate_hands = jax.vmap(generate_hands, out_axes=0)
        h1_twohots, h2_twohots, sp_twohots, = v_generate_hands(rngs)
        return (sp_twohots, h1_twohots, h2_twohots)


    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, target, guess):
        condition = jax.lax.eq(target, guess) # elementwise compare
        result = jnp.all(condition, axis=-1) # apply AND logic on the "two-hot" axis
        return jnp.where(result, 1, 0)