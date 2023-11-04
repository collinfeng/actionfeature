import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import flax
import flax.linen as nn

from utils.utils import create_train_state


class single_head_SA2IAttn(nn.Module):
    qkv_features: int
    batch_size: int
    seq_len: int

    def setup(self):
        self.Q = nn.Dense(self.qkv_features)
        # self.K = nn.Dense(self.qkv_features)
        # self.V = nn.Dense(self.qkv_features)

    def mult_dot(self, q, k, v):
        similarities = jnp.einsum("bif, bjf->bij", q, k)
        attention_prob = nn.softmax(similarities/jnp.sqrt(self.seq_len), axis=-1)
        attention_out = jnp.einsum("bij, bjf->bif", attention_prob, v)
        return attention_out

    def __call__(self, x):
        q = self.Q(x.reshape(-1, self.qkv_features)).reshape(self.batch_size, -1, self.qkv_features)
        # k = self.K(x.reshape(-1, self.qkv_features)).reshape(self.batch_size, -1, self.qkv_features)
        # v = self.V(x.reshape(-1, self.qkv_features)).reshape(self.batch_size, -1, self.qkv_features)
        attention = self.mult_dot(q, q, q)
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

    def observation_shaping(self, sp, h1, h2):
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
    
class SA2I(nn.Module):
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
        self.SA2IAttn = single_head_SA2IAttn(qkv_features=self.qkv_features, batch_size=self.batch_size, seq_len=(2 * self.N + 2))

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
            x = self.SA2IAttn(x)
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
        q_values = vmap_forward_pass(observation, actions)
        return q_values.reshape(self.batch_size, -1)
    

def model_test():
    model  = SA2I(hidden=128,
                 num_heads=1,
                 batch_size=500,
                 emb_dim=9,
                 N=5,
                 qkv_features=9,
                 out_features=9)
    init_sp = jnp.zeros((500, 6), jnp.float32)
    init_h1 = jnp.zeros((500, 5, 6), jnp.float32)
    init_h2 = jnp.zeros((500, 5, 6), jnp.float32)
    t_state = create_train_state(model, init_sp, init_h1, init_h2, jax.random.PRNGKey(0), 0.1)
    q_values = t_state.apply_fn({"params": t_state.params}, init_sp, init_h1, init_h2)
    print(q_values.shape)

