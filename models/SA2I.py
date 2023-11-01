import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import flax
import flax.linen as nn

from utils.utils import create_train_state


class single_head_SA2IAttn(nn.Module):
    qkv_features: int
    batch_size: int

    def setup(self):
        self.W = nn.Dense(self.qkv_features)

    def mult_dot(self, x):
        similarities = jnp.einsum("ijk, isk->ijs", x, x)
        attention_prob = nn.softmax(similarities, axis=-1)/jnp.sqrt(self.qkv_features)
        attention_out = jnp.einsum("ijk, iks->ijs", attention_prob, x)
        return attention_out

    def __call__(self, x):
        x = self.W(x.reshape(-1, self.qkv_features)).reshape(self.batch_size, -1, self.qkv_features)
        attention = self.mult_dot(x)
        return attention


class A2I(nn.Module):
    hidden: int
    num_heads: int
    batch_size: int
    emb_dim: int
    N: int
    qkv_features: int
    out_features: int

    def setup(self):
        self.fc1 = nn.Dense(self.hidden)
        self.fc2 = nn.Dense(self.hidden)
        self.fc3 = nn.Dense(1)
        self.attn = nn.SelfAttention(self.num_heads, qkv_features=self.qkv_features, out_features=self.out_features)
        self.SA2IAttn = single_head_SA2IAttn(qkv_features=self.qkv_features, batch_size=self.batch_size)

    def observation_shaping(self, sp, h1, h2):
        # sp = jnp.concatenate((jnp.repeat(jnp.array([1, 0, 0]), self.batch_size).reshape(self.batch_size, 3), sp), axis=-1)
        # h1 = jnp.concatenate((jnp.repeat(jnp.array([0, 1, 0]), self.batch_size * self.N).reshape(self.batch_size, self.N, 3), h1), axis=-1)
        # h2 = jnp.concatenate((jnp.repeat(jnp.array([0, 1, 0]), self.batch_size * self.N).reshape(self.batch_size, self.N, 3), h2), axis=-1)
        result = jnp.concatenate((sp[:, jnp.newaxis, :], h1, h2), axis=1)
        return result

    def actions_shaping(self, h2):
        return jnp.concatenate((jnp.repeat(jnp.array([0, 0, 1]), self.batch_size * self.N).reshape(self.batch_size, self.N, 3), h2), axis=-1)

    def __call__(self, sp, h1, h2):
        def forward_pass(observation, action):
            x = jnp.concatenate((observation, action[:, jnp.newaxis, :]), axis=1)
            x = self.attn(x)
            x = jnp.mean(x, axis=1)
            x = self.fc1(x)
            x = nn.relu(x)
            x = self.fc2(x)
            x = nn.relu(x)
            x = self.fc3(x)
            
            return x
        vmap_forward_pass = jax.vmap(forward_pass, in_axes=(None, 1), out_axes=1)
        observation = self.observation_shaping(sp, h1, h2)
        # actions = self.actions_shaping(h2)
        q_values = vmap_forward_pass(observation, h2).reshape(self.batch_size, -1)
        return q_values
    
class A2ICoded(nn.Module):
    hidden: int
    num_heads: int
    batch_size: int
    emb_dim: int
    N: int
    qkv_features: int
    out_features: int

    def setup(self):
        self.fc1 = nn.Dense(self.hidden)
        self.fc2 = nn.Dense(self.hidden)
        self.fc3 = nn.Dense(1)
        self.attn = nn.SelfAttention(self.num_heads, qkv_features=self.qkv_features, out_features=self.out_features)
        self.SA2IAttn = single_head_SA2IAttn(qkv_features=self.qkv_features, batch_size=self.batch_size)

    def observation_shaping(self, sp, h1, h2):
        sp = jnp.concatenate((jnp.repeat(jnp.array([1, 0, 0]), self.batch_size).reshape(self.batch_size, 3), sp), axis=-1)
        h1 = jnp.concatenate((jnp.repeat(jnp.array([0, 1, 0]), self.batch_size * self.N).reshape(self.batch_size, self.N, 3), h1), axis=-1)
        h2 = jnp.concatenate((jnp.repeat(jnp.array([0, 1, 0]), self.batch_size * self.N).reshape(self.batch_size, self.N, 3), h2), axis=-1)
        result = jnp.concatenate((sp[:, jnp.newaxis, :], h1, h2), axis=1)
        return result

    def actions_shaping(self, h2):
        return jnp.concatenate((jnp.repeat(jnp.array([0, 0, 1]), self.batch_size * self.N).reshape(self.batch_size, self.N, 3), h2), axis=-1)

    def __call__(self, sp, h1, h2):
        def forward_pass(observation, action):
            x = jnp.concatenate((observation, action[:, jnp.newaxis, :]), axis=1)
            x = self.attn(x)
            x = jnp.mean(x, axis=1)
            x = self.fc1(x)
            x = nn.relu(x)
            x = self.fc2(x)
            x = nn.relu(x)
            x = self.fc3(x)
            
            return x
        vmap_forward_pass = jax.vmap(forward_pass, in_axes=(None, 1), out_axes=1)
        observation = self.observation_shaping(sp, h1, h2)
        actions = self.actions_shaping(h2)
        q_values = vmap_forward_pass(observation, actions).reshape(self.batch_size, -1)
        return q_values

def model_test():
    model  = A2I(hidden=128,
                 num_heads=1,
                 batch_size=256,
                 emb_dim=6,
                 N=5,
                 qkv_features=6,
                 out_features=6)
    init_sp = jnp.zeros((256, 6), jnp.float32)
    init_h1 = jnp.zeros((256, 5, 6), jnp.float32)
    init_h2 = jnp.zeros((256, 5, 6), jnp.float32)
    t_state = create_train_state(model, init_sp, init_h1, init_h2, jax.random.PRNGKey(0), 0.1)
    q_values = t_state.apply_fn({"params": t_state.params}, init_sp, init_h1, init_h2)
    print(q_values.shape)

