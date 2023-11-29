import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

from models.hg_models import *
from environments.hintguess import *
from utils.utils import *
from utils.evaluations import *

def train_agents_dropout(config):
    # jitted later jitted later after vmapped
    def train_sigle_agent(rngs, t_state_h, t_state_g, eps):
        
        def eps_policy(config, eps, q_values, rng):
                def rand_action(dummy):
                    return jax.random.randint(rng, (), 0, config["N"])
                def greedy(dummy):
                    return jnp.argmax(q_values)
                condition = jax.numpy.less_equal(jax.random.uniform(rng), eps)
                return jax.lax.cond(condition, (), rand_action, (), greedy)
        
        def training_step(carry, x):
            def loss_fn(h_params, g_params, rng, eps):
                rng, h_rng, g_rng, env_rng = jax.random.split(rng, 4)
                tgt_twohot, H1_twohot, H2_twohot = hg_env.get_observation(env_rng)

                # q_values_h = t_state_h.apply_fn({"params": h_params}, tgt_twohot, H2_twohot, H1_twohot)
                q_values_h = t_state_h.apply_fn({"params": h_params}, tgt_twohot, H2_twohot, H1_twohot, training=False, rngs={'dropout': h_rng})

                rngs = jax.random.split(h_rng, batch_size)
                h_actions = eps_v(config, eps, q_values_h, rngs)
                q_h = jnp.take_along_axis(q_values_h, h_actions[:, jnp.newaxis], axis=1).squeeze(axis=1)
                hinted_twohot = jnp.take_along_axis(H1_twohot, h_actions[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
                # q_values_g = t_state_g.apply_fn({"params": g_params}, hinted_twohot, H1_twohot, H2_twohot)
                q_values_g = t_state_g.apply_fn({"params": g_params}, hinted_twohot, H1_twohot, H2_twohot, training=False, rngs={'dropout': g_rng})

                rngs = jax.random.split(g_rng, batch_size)
                guess = eps_v(config, eps, q_values_g, rngs)
                q_g = jnp.take_along_axis(q_values_g, guess[:, jnp.newaxis], axis=1).squeeze(axis=1)
                guess_twohot = jnp.take_along_axis(H2_twohot, guess[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
                rewards = hg_env.get_reward(tgt_twohot, guess_twohot)

                h_loss = jnp.mean((q_h - rewards)**2)
                g_loss = jnp.mean((q_g - rewards)**2)
                # jax.debug.print()

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
            # jax.debug.print("🤯 {x} 🤯", x=grad_h['test3'])
            grad_g_loss_fn = jax.grad(g_loss_fn)
            grad_g = grad_g_loss_fn(t_state_g.params, rng, eps)
            t_state_g = t_state_g.apply_gradients(grads = grad_g)
            rewards = calc_rewards(rng, eps)
            return (t_state_h, t_state_g), rewards
        
        eps_v = jax.vmap(eps_policy, in_axes=(None, None, 0, 0))
        (t_state_h, t_state_g), rewards = jax.lax.scan(training_step, (t_state_h, t_state_g), (rngs, eps))

        return t_state_h, t_state_g, rewards

    # later vmapped
    def init_train_states(init_rng):
        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], dropout_rng=init_rng)
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], dropout_rng=init_rng)
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
    model = config["model"]

    init_sp = jnp.zeros((config["batch_size"], 2 * config["feature_dim"]), jnp.float32)
    init_h1 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
    init_h2 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
    train_rng = jax.random.PRNGKey(config["train_rng"])
    init_rng = jax.random.PRNGKey(config["init_rng"])

    eps_min = config["eps_min"]
    eps_max = config["eps_max"]
    K = config["K"]
    eval_interval = config["eval_interval"]
    hinter = model(config["mlp_hidden"],
                config["num_heads"],
                config["batch_size"],
                config["emb_dim"],
                config["N"],
                config["qkv_features"],
                config["out_features"],
                config["dropout"])
        
    guesser = model(config["mlp_hidden"],
                    config["num_heads"],
                    config["batch_size"],
                    config["emb_dim"],
                    config["N"],
                    config["qkv_features"],
                    config["out_features"],
                    config["dropout"])

    # batched training setup
    # setup rngs and eps
    num_evals = num_episodes // eval_interval
    train_rngs = jax.random.split(train_rng, num_episodes * num_agents).reshape(num_evals, num_agents, -1, 2)
    init_rngs = jax.random.split(init_rng, num_agents)
    n = jnp.arange(num_episodes)
    eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)
    eps = eps.reshape(num_evals, -1)
   
    # init batched train_state
    batch_init = jax.vmap(init_train_states, in_axes=(0,))
    batch_t_state_h, batch_t_state_g = batch_init(init_rngs) 

    # batched_train
    batch_train = jax.vmap(train_sigle_agent, in_axes=(0, 0, 0, None,), out_axes=0)
    jitted_batch_train = jax.jit(batch_train)

    # logging
    batch_rewards = np.zeros((num_agents, num_evals, eval_interval))
    xp_scores = np.zeros(num_evals)
    
    for eval_idx in range(num_evals):
        batch_interval_train_rngs = train_rngs[eval_idx, :, :, :]
        interval_eps = eps[eval_idx, :]
        batch_t_state_h, batch_t_state_g, batch_interval_reward = jitted_batch_train(batch_interval_train_rngs, batch_t_state_h, batch_t_state_g, interval_eps)
        batch_rewards[:, eval_idx, :] = batch_interval_reward
        xp_scores[eval_idx] = batched_xp_eval_drop_out(batch_t_state_h, batch_t_state_g, config).mean()
        print(f"Episode: {(1 + eval_idx) * eval_interval} sp_train_scores: {batch_interval_reward.mean()}, xp_train_scores: {xp_scores[eval_idx]}")

    xp_train_scores = xp_scores.reshape(-1)
    sp_train_scores = batch_rewards.reshape(num_agents, -1)
    
    return batch_t_state_h, batch_t_state_g, sp_train_scores, xp_train_scores
    

def train_agents(config):
    # jitted later jitted later after vmapped
    def train_sigle_agent(rngs, t_state_h, t_state_g, eps):
        
        def eps_policy(config, eps, q_values, rng):
                def rand_action(dummy):
                    return jax.random.randint(rng, (), 0, config["N"])
                def greedy(dummy):
                    return jnp.argmax(q_values)
                condition = jax.numpy.less_equal(jax.random.uniform(rng), eps)
                return jax.lax.cond(condition, (), rand_action, (), greedy)
        
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
        
        eps_v = jax.vmap(eps_policy, in_axes=(None, None, 0, 0))
        (t_state_h, t_state_g), rewards = jax.lax.scan(training_step, (t_state_h, t_state_g), (rngs, eps))

        return t_state_h, t_state_g, rewards

    # jitted later jitted later after vmapped
    def init_train_states(init_rng):
        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
        return t_state_h, t_state_g


    '''
    Current function: train_agents(config, cp_suffix):
    Declare static variables
    '''
    hg_env = HintGuessEnv(config)
    num_agents = config["num_agents"]
    batch_size = config["batch_size"]
    num_episodes = config["num_episodes"]
    train_rng = jax.random.PRNGKey(config["train_rng"])
    init_rng = jax.random.PRNGKey(config["init_rng"])
    eps_min = config["eps_min"]
    eps_max = config["eps_max"]
    K = config["K"]
    eval_interval = config["eval_interval"]

    init_sp, init_h1, init_h2, hinter, guesser = init_model(config)

    train_rngs = jax.random.split(train_rng, num_episodes * num_agents).reshape(num_agents, -1, 2)
    init_rngs = jax.random.split(init_rng, num_agents)
    n = jnp.arange(num_episodes)
    eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)
   
    # init batched train_state
    batch_init = jax.jit(jax.vmap(init_train_states, in_axes=(0,)))
    batch_t_state_h, batch_t_state_g = batch_init(init_rngs) 

    # batched_train
    batch_train = jax.vmap(train_sigle_agent, in_axes=(0, 0, 0, None,), out_axes=0)
    jitted_batch_train = jax.jit(batch_train)

    # logging
    batch_rewards = np.zeros((num_agents, eval_interval))
    batch_t_state_h, batch_t_state_g, batch_reward = jitted_batch_train(train_rngs, batch_t_state_h, batch_t_state_g, eps)
    

    xp_train_scores = np.zeros()
    sp_train_scores = batch_rewards.reshape(num_agents, -1)
    
    return batch_t_state_h, batch_t_state_g, sp_train_scores, xp_train_scores