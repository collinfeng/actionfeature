import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

from models.SA2I import *
from environments.hintguess import *
from utils.utils import *
from utils.eval import *


def train_agents(config, cp_suffix):
    
    # @jax.jit
    # def batched_xp_eval(batch_t_state_h, batch_t_state_g):
    #     def 
    
    @jax.jit
    def train_sigle_agent(rngs, t_state_h, t_state_g, eps):
        def training_step(carry, x):
            def loss_fn(h_params, g_params, rng, eps):

                rng, subrng = jax.random.split(rng)
                tgt_twohot, H1_twohot, H2_twohot = hg_env.get_observation(subrng)

                q_values_h = t_state_h.apply_fn({"params": h_params}, tgt_twohot, H2_twohot, H1_twohot)

                rng, subrng = jax.random.split(rng)
                rngs = jax.random.split(subrng, batch_size)
                h_actions = eps_v(config, eps, q_values_h, rngs)
                q_h = jnp.take_along_axis(q_values_h, h_actions[:, jnp.newaxis], axis=1).squeeze(axis=1)
                hinted_twohot = jnp.take_along_axis(H1_twohot, h_actions[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
                q_values_g = t_state_g.apply_fn({"params": g_params}, hinted_twohot, H1_twohot, H2_twohot)

                rng, subrng = jax.random.split(rngs[-1])
                rngs = jax.random.split(subrng, batch_size)
                guess = eps_v(config, eps, q_values_g, rngs)
                q_g = jnp.take_along_axis(q_values_g, guess[:, jnp.newaxis], axis=1).squeeze(axis=1)
                guess_twohot = jnp.take_along_axis(H2_twohot, guess[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
                rewards = hg_env.get_reward(tgt_twohot, guess_twohot)

                h_loss = jnp.mean((q_h - rewards)**2)
                g_loss = jnp.mean((q_g - rewards)**2)

                return h_loss, g_loss, jnp.mean(rewards)

            def h_loss_fn(h_params, rng, eps):
                h_loss, _, _ = loss_fn(h_params, t_state_g.params, rng, eps)
                return h_loss

            def g_loss_fn(g_params, rng, eps):
                _, g_loss, _ = loss_fn(t_state_h.params, g_params, rng, eps)
                return g_loss

            def calc_rewards(rng, eps):
                _, _, rewards = loss_fn(t_state_h.params, t_state_g.params, rng, eps)
                return rewards

            rng, eps = x
            t_state_h, t_state_g = carry
            grad_h_loss_fn = jax.grad(h_loss_fn)
            grad_h = grad_h_loss_fn(t_state_h.params, rng, eps)
            t_state_h = t_state_h.apply_gradients(grads = grad_h)
            grad_g_loss_fn = jax.grad(g_loss_fn)
            grad_g = grad_g_loss_fn(t_state_g.params, rng, eps)
            t_state_g = t_state_g.apply_gradients(grads = grad_g)
            rewards = calc_rewards(rng, eps)
            return (t_state_h, t_state_g), rewards
        
        (t_state_h, t_state_g), rewards = jax.lax.scan(training_step, (t_state_h, t_state_g), (rngs, eps))

        return t_state_h, t_state_g, rewards

    @jax.jit
    def init_train_states(init_rng):
        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
        return t_state_h, t_state_g


    '''
    Current function: train_agents(config, cp_suffix):
    Declare static variables
    '''
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
    train_rngs = jax.random.split(train_rng, num_episodes * num_agents).reshape(eval_interval, num_agents, -1, 2)
    init_rngs = jax.random.split(init_rng, num_agents)
    n = jnp.arange(num_episodes)
    eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)
    eps = eps.reshape(eval_interval, -1)
    num_evals = num_episodes // eval_interval


    # init batched train_state
    batch_init = jax.vmap(init_train_states, in_axes=(0,))
    batch_t_state_h, batch_t_state_g = batch_init(init_rngs) 


    # batched_train
    batch_train = jax.vmap(train_sigle_agent, in_axes=(0, 0, 0, None,), out_axes=0)

    # logging
    batch_rewards = []
    xp_scores = []
    
    for eval_idx in range(num_evals):
        batch_interval_train_rngs = train_rngs[eval_idx, :, :]
        interval_eps = eps[eval_idx, :]
        batch_t_state_h, batch_t_state_g, batch_interval_reward = batch_train(batch_interval_train_rngs, batch_t_state_h, batch_t_state_g, interval_eps)
        hinters = [jax.tree_map(lambda x: x[i], batch_t_state_h) for i in range(num_agents)]
        guessers = [jax.tree_map(lambda x: x[i], batch_t_state_h) for i in range(num_agents)]
        agents = [[hinters[i], guessers[i]] for i in range(num_agents)]
        xp_scores.append(xp_eval(agents, config))
        batch_rewards.append(batch_interval_reward)

    xp_train_scores = jnp.stack(xp_scores)
    sp_train_scores = jnp.stack(batch_rewards)
    
    # save result
    if config["save_result"] == True:
        save_batched_pytree(batch_t_state_h, f"checkpoints/{currentDate}-{cp_suffix}/hinter", num_agents)
        save_batched_pytree(batch_t_state_g, f"checkpoints/{currentDate}-{cp_suffix}/guesser", num_agents)
        if not os.path.isdir("results/{currentDate}-{cp_suffix}"):
            os.mkdir(f"results/{currentDate}-{cp_suffix}")
        save_jax_array(sp_train_scores, f"results/{currentDate}-{cp_suffix}", "sp_train_scores")
        save_jax_array(xp_train_scores, f"results/{currentDate}-{cp_suffix}", "xp_train_scores")
    
