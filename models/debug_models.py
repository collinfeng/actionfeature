import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import flax
import flax.linen as nn
from jax.nn import initializers

from utils.utils import create_train_state

class Norm(nn.Module):
    seq_len: int
    emb_dim: int
    eps: float = 1e-6
    '''
    The param definition suits input x of dim batch, features, cards
    '''
    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', initializers.ones, (self.emb_dim, self.seq_len))
        bias = self.param('bias', initializers.zeros, (self.emb_dim, self.seq_len))

        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        norm = alpha * (x - mean) / (std + self.eps) + bias
        return norm

class Attn3d(nn.Module):
    seq_len: int
    qkv_dim: int
    drop_out: float = 0.1

    def setup(self):
        self.q_linear = nn.Dense(self.seq_len)
        self.k_linear = nn.Dense(self.seq_len)
        self.v_linear = nn.Dense(self.seq_len)

        self.norm_1 = Norm(self.seq_len, self.qkv_dim)
        self.norm_2 = Norm(self.seq_len, self.qkv_dim)

        self.drop_out_1 = nn.Dropout(rate=self.drop_out)
        self.drop_out_2 = nn.Dropout(rate=self.drop_out)

    def __call__(self, x, training=False):
        '''
        x: batch, features, cards
        '''
        x = self.norm_1(x)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        similarity = jnp.einsum("bif, bjf->bij", q, k)
        attention_prob = nn.softmax(similarity/jnp.sqrt(self.seq_len), axis=-1)
        attention_prob = self.drop_out_1(attention_prob, deterministic=not training)
        attention_out = jnp.einsum("bij, bjf->bif", attention_prob, v)
        attention_out = self.norm_2(attention_out + self.drop_out_2(attention_out, deterministic=not training))
        return attention_out, similarity, q, k, v
    
class AttnModel3d(nn.Module):
    hidden: int
    num_heads: int
    batch_size: int
    emb_dim: int
    N: int
    qkv_features: int
    out_features: int
    drop_out: float

    def setup(self):
        # remarks: any definition without using nn will not be trained
        self.attn = Attn3d(self.N * 2 + 2, self.qkv_features, self.drop_out)
        self.linear = nn.Dense(1)
        

    def observation_shaping(self, sp, h1, h2):
        obs = jnp.concatenate((h1, h2, sp[:, jnp.newaxis, :]), axis=1)
        return obs
    
    def forward_pass(self, observation, action):
            x = jnp.concatenate((observation, action[:, :, jnp.newaxis]), axis=-1)
            attn_out, sim, q, k, v = self.attn(x)
            x = attn_out.reshape(self.batch_size, -1)
            x = self.linear(x)
            return x, sim, attn_out, q, k, v

    def __call__(self, sp, h1, h2, training=False):
        '''
        h1, h2: batch, cards, features
        sp: batch, features
        
        Notes: 
        to aline with the attn3 definition, we need to transpose x to batch, features, cards
        same for h2
        '''
        obs = self.observation_shaping(sp, h1, h2)
        obs = obs.transpose((0, 2, 1))
        h2 = h2.transpose((0, 2, 1))
        forward_pass_v = jax.vmap(self.forward_pass, in_axes=(None, -1), out_axes=(1, 1, 1, 1, 1, 1))
        q_values, similarities, attn_out, q, k, v = forward_pass_v(obs, h2)
        q_values = q_values.reshape(self.batch_size, self.N)
        q = q.reshape(self.batch_size, self.N, self.emb_dim, self.N * 2 + 2)
        k = k.reshape(self.batch_size, self.N, self.emb_dim, self.N * 2 + 2)
        v = v.reshape(self.batch_size, self.N, self.emb_dim, self.N * 2 + 2)
        similarities = similarities.reshape(self.batch_size, self.N, self.emb_dim, self.emb_dim)
        attn_out = attn_out.reshape(self.batch_size, self.N, self.emb_dim, self.N * 2 + 2)
        
        
        
        return q_values, similarities, attn_out, q, k, v


class Normv(nn.Module):
    seq_len: int
    emb_dim: int
    eps: float = 1e-6
    '''
    The param definition suits input x of dim batch, features, cards
    '''
    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', initializers.ones, (self.seq_len, self.emb_dim))
        bias = self.param('bias', initializers.zeros, (self.seq_len, self.emb_dim))

        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        norm = alpha * (x - mean) / (std + self.eps) + bias
        return norm

class Attn3vd(nn.Module):
    seq_len: int
    qkv_dim: int
    drop_out: float = 0.1

    def setup(self):
        self.q_linear = nn.Dense(self.qkv_dim)
        self.k_linear = nn.Dense(self.qkv_dim)
        self.v_linear = nn.Dense(self.qkv_dim)

        self.norm_1 = Normv(self.seq_len, self.qkv_dim)
        self.norm_2 = Normv(self.seq_len, self.qkv_dim)

        self.drop_out_1 = nn.Dropout(rate=self.drop_out)
        self.drop_out_2 = nn.Dropout(rate=self.drop_out)

    def __call__(self, x, training=False):
        '''
        x: batch, features, cards
        '''
        x = self.norm_1(x)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        similarity = jnp.einsum("bif, bjf->bij", q, k)
        attention_prob = nn.softmax(similarity/jnp.sqrt(self.seq_len), axis=-1)
        attention_prob = self.drop_out_1(attention_prob, deterministic=not training)
        attention_out = jnp.einsum("bij, bjf->bif", attention_prob, v)
        attention_out = self.norm_2(attention_out + self.drop_out_2(attention_out, deterministic=not training))
        return attention_out, similarity, q, k, v
    
class AttnModel3vd(nn.Module):
    hidden: int
    num_heads: int
    batch_size: int
    emb_dim: int
    N: int
    qkv_features: int
    out_features: int
    drop_out: float

    def setup(self):
        # remarks: any definition without using nn will not be trained
        self.attn = Attn3vd(self.N * 2 + 2, self.qkv_features, self.drop_out)
        self.linear = nn.Dense(1)
        

    def observation_shaping(self, sp, h1, h2):
        obs = jnp.concatenate((h1, h2, sp[:, jnp.newaxis, :]), axis=1)
        return obs
    
    def forward_pass(self, observation, action):
            x = jnp.concatenate((observation, action[:, jnp.newaxis, :]), axis=1)
            attention_out, similarity, q, k, v = self.attn(x)
            x = attention_out.reshape(self.batch_size, -1)
            x = self.linear(x)
            return x, similarity, attention_out, q, k, v

    def __call__(self, sp, h1, h2, training=False):
        '''
        h1, h2: batch, cards, features
        sp: batch, features
        
        Notes: 
        to aline with the attn3 definition, we need to transpose x to batch, features, cards
        same for h2
        '''
        obs = self.observation_shaping(sp, h1, h2)
        forward_pass_v = jax.vmap(self.forward_pass, in_axes=(None, 1), out_axes=(1, 1, 1, 1, 1, 1))
        q_values, similarities, attn_out, q, k, v = forward_pass_v(obs, h2)

        q_values = q_values.reshape(self.batch_size, self.N)
        q = q.reshape(self.batch_size, self.N, self.N * 2 + 2, self.emb_dim)
        k = k.reshape(self.batch_size, self.N, self.N * 2 + 2, self.emb_dim)
        v = v.reshape(self.batch_size, self.N, self.N * 2 + 2, self.emb_dim)
        similarities = similarities.reshape(self.batch_size, self.N, self.N * 2 + 2, self.N * 2 + 2)
        attn_out = attn_out.reshape(self.batch_size, self.N, self.N * 2 + 2, self.emb_dim)
        return q_values, similarities, attn_out, q, k, v