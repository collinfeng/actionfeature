import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from tqdm import tqdm

from models.SA2I import *
from environments.hintguess import *
from utils.utils import *


def train_agents(config, cp_suffix):
    '''
    declare static arguments
    '''
    num_agents = config["num_agents"]
    batch_size = config["batch_size"]
    num_episodes = config["num_episodes"]
    N = config["N"]

    static_args = (num_agents, batch_size, num_episodes, N)
    currentDate = datetime.now().strftime("%Y-%m-%d")
    
    
    @jax.jit
    def a2i_train(rng):
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

            # jax.debug.print("🤯 {x} 🤯", x=rewards.mean())

            return (t_state_h, t_state_g), rewards

        num_of_objects = 2*N + 2
        
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
        
        
        init_sp = jnp.zeros((config["batch_size"], 2 * config["feature_dim"]), jnp.float32)
        init_h1 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
        init_h2 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
        init_rng = rng
        # init_rng = jax.random.PRNGKey(12345)
        hg_env = HintGuessEnv(config)
        eps_v = jax.vmap(eps_policy, in_axes=(None, None, 0, 0))
        eps_min = config["eps_min"]
        eps_max = config["eps_max"]
        K = config["K"]

        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"])
        rngs = jax.random.split(rng, num_episodes)
        n = jnp.arange(num_episodes)
        eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)
        # jax.debug.print("🤯 {x} 🤯", x=eps)


        (t_state_h, t_state_g), rewards = jax.lax.scan(training_step, (t_state_h, t_state_g), (rngs, eps))

        info = rewards
        return t_state_h, t_state_g, info

    num_agents, batch_size, num_episodes, N = static_args


    rng = jax.random.PRNGKey(config["PRNGkey"])
    rngs = jax.random.split(rng, num_agents)
    batch_train = jax.vmap(a2i_train, in_axes=(0,))
    batch_t_state_h, batch_t_state_g, batch_rewards = batch_train(rngs)
    save_batched_pytree(batch_t_state_h, f"checkpoints/{currentDate}-{cp_suffix}/hinter", num_agents)
    save_batched_pytree(batch_t_state_g, f"checkpoints/{currentDate}-{cp_suffix}/guesser", num_agents)
    save_jax_array(batch_rewards, f"results/{currentDate}-{cp_suffix}", "rewards")
    
