import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from tqdm import tqdm
from flax.training import train_state, checkpoints

from models.debug_models import *
from environments.hintguess import *
from utils.utils import *
from utils.evaluations import *

import matplotlib.pyplot as plt


def test_tx():
    config = {
        "debug":True,
        "N": 5,
        "feature_dim": 3, # this the number of classes under different features set, e.g. dim = 3 for 0, 1, 2
        "emb_dim":9,
        "qkv_features":9,
        "out_features":9,
        "num_episodes": 800000,
        "mlp_hidden": 128,
        "batch_size": 500,
        "learning_rate": 0.0001,
        "num_agents": 4,
        "init_rng": 235711,
        "train_rng": 54321,
        "eval_rng": 12345,
        "num_heads": 1,
        "eval_runs":10,
        "eps_min":0.01,
        "eps_max":0.95,
        "K":50000,
        "logging":True,
        "eval_interval":2000,
        "save_result": True
        }

    num_agents = config["num_agents"]
    batch_size = config["batch_size"]
    num_episodes = config["num_episodes"]
    N = config["N"]
    hg_env = HintGuessEnv(config)

    currentDate = datetime.now().strftime("%Y-%m-%d")
    init_sp = jnp.zeros((config["batch_size"], 2 * config["feature_dim"]), jnp.float32)
    init_h1 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
    init_h2 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
    train_rng = jax.random.PRNGKey(config["train_rng"])
    init_rng = jax.random.PRNGKey(config["init_rng"])

    eps_v = jax.vmap(eps_policy, in_axes=(None, None, 0, 0))
    eps_min = config["eps_min"]
    eps_max = config["eps_max"]
    K = config["K"]
    eval_interval = config["eval_interval"]

    hinter = A2ICoded(hidden=config["mlp_hidden"],
                    num_heads=config["num_heads"],
                    batch_size=config["batch_size"],
                    emb_dim=config["emb_dim"],
                    N=config["N"],
                    qkv_features=config["qkv_features"],
                    out_features=config["out_features"])
        
    guesser = A2ICoded(hidden=config["mlp_hidden"],
                    num_heads=config["num_heads"],
                    batch_size=config["batch_size"],
                    emb_dim=config["emb_dim"],
                    N=config["N"],
                    qkv_features=config["qkv_features"],
                    out_features=config["out_features"])

    # batched training setup
    # setup rngs and eps
    train_rngs = jax.random.split(train_rng, num_episodes * num_agents).reshape(num_agents, eval_interval, -1)
    init_rngs = jax.random.split(init_rng, num_agents)
    n = jnp.arange(num_episodes)
    eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)
    eps = eps.reshape(eval_interval, -1)
    num_evals = num_episodes / eval_interval

    def init_train_states(init_rng):
            t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
            t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
            return t_state_h, t_state_g

    # batched training setup
    # setup rngs and eps
    train_rngs = jax.random.split(train_rng, num_episodes * num_agents).reshape(num_agents, eval_interval, -1)
    init_rngs = jax.random.split(init_rng, num_agents)
    n = jnp.arange(num_episodes)
    eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)
    eps = eps.reshape(eval_interval, -1)
    num_evals = num_episodes / eval_interval

    # init batched train_state
    batch_init = jax.vmap(init_train_states, in_axes=(0,))
    batch_t_state_h, batch_t_state_g = batch_init(init_rngs)

    
   

def test_seed():
    a = jax.random.PRNGKey(0)
    print(a.shape)

@jax.jit
def test_transpose():
    a = jnp.arange(8).reshape(2, 2, 2)
    shape = np.array(a.shape)
    shape[0] = shape[0] * 2
    axes = np.concatenate((np.arange(a.ndim) + 1, [0]), axis=-1)
    a = jnp.repeat(a[jnp.newaxis, ...], 2, axis=0)
    print(axes)
    a = jnp.transpose(a, axes=axes)
    a = a.reshape(shape)
    return a



rewards = np.load("results/2023-11-02-SA2I-2MLP/sp_train_scores")
rewards.shape
plt.plot(rewards[0])