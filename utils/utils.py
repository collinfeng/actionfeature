import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import orbax.checkpoint as ocp
from tempfile import TemporaryFile
import numpy as np

def create_train_state(model, init_sp, init_h1, init_h2, init_rng, lr, params=None):
    optim = optax.adam(lr)
    if params != None:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optim)
    else:
        params = model.init({"params": init_rng}, init_sp, init_h1, init_h2)["params"]
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optim)


def save_pytree(pytree, path):
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, pytree)

def load_trainstate(path):
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path)

def save_batched_pytree(batched_pytree, path, n):
    def index_array(i, x):
        return x[i]
    for i in range(n):
        py_tree = jax.tree_map(lambda x: index_array(i, x), batched_pytree)
        save_pytree(py_tree, path + f"_{i}")

def save_jax_array(ndarray, path="result", filename="untitled.npy"):
    with open(f"{path}/{filename}", 'wb') as f:
        jnp.save(f, ndarray)

@jax.jit
def eps_policy(config, eps, q_values, rng):

    def rand_action(dummy):
        return jax.random.randint(rng, (), 0, config["N"])

    def greedy(dummy):
        return jnp.argmax(q_values)

    condition = jax.numpy.less_equal(jax.random.uniform(rng), eps)
    return jax.lax.cond(condition, (), rand_action, (), greedy)



 
