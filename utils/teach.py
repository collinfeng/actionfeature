import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import orbax.checkpoint as ocp
from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot as plt
from models.hg_models import *
from utils.utils import *
from utils.evaluations import *
from flax.training import train_state

def teach_agents(config, teachers):
    '''
    config: dict of student's config for training
    teachers: tuple of batched hinter/guesser tx states
    '''

    # jitted later jitted later after vmapped
    def train_sigle_agent(rngs, t_state_h, t_state_g, teacher_hinter, teacher_guesser, eps):
        
        def eps_policy(config, eps, q_values, rng):
            def rand_action(dummy):
                return jax.random.randint(rng, (), 0, config["N"])
            def greedy(dummy):
                return jnp.argmax(q_values)
            eps = 0 # force greedy policy, no exploration is requried as we mirror the policy of teacher directly
            condition = jax.numpy.less_equal(jax.random.uniform(rng), eps)
            return jax.lax.cond(condition, (), rand_action, (), greedy)
        
        def training_step(carry, x):
            def loss_fn(h_params, g_params, rng, eps):
                rng, h_rng, g_rng, env_rng = jax.random.split(rng, 4)
                tgt_twohot, H1_twohot, H2_twohot = hg_env.get_observation(env_rng)

                # teacher & student hinter provide their estimate to Q values
                q_values_h_teacher = teacher_hinter.apply_fn({"params": teacher_hinter.params}, tgt_twohot, H2_twohot, H1_twohot)
                q_values_h = t_state_h.apply_fn({"params": h_params}, tgt_twohot, H2_twohot, H1_twohot, training=False, rngs={'dropout': h_rng})


                rngs = jax.random.split(h_rng, batch_size)
                h_actions = eps_v(config, eps, q_values_h_teacher, rngs)
                # q_h = jnp.take_along_axis(q_values_h_teacher, h_actions[:, jnp.newaxis], axis=1).squeeze(axis=1)
                hinted_twohot = jnp.take_along_axis(H1_twohot, h_actions[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)

                # teacher & student guesser provide their estimate to Q values
                q_values_g_teacher = teacher_guesser.apply_fn({"params": teacher_guesser.params}, hinted_twohot, H1_twohot, H2_twohot)
                q_values_g = t_state_g.apply_fn({"params": g_params}, hinted_twohot, H1_twohot, H2_twohot, training=False, rngs={'dropout': g_rng})

                rngs = jax.random.split(g_rng, batch_size)
                guess = eps_v(config, eps, q_values_g_teacher, rngs)
                # q_g = jnp.take_along_axis(q_values_g, guess[:, jnp.newaxis], axis=1).squeeze(axis=1)
                guess_twohot = jnp.take_along_axis(H2_twohot, guess[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
                rewards = hg_env.get_reward(tgt_twohot, guess_twohot)
                
                h_loss = jnp.mean((q_values_h - q_values_h_teacher)**2)
                g_loss = jnp.mean((q_values_g - q_values_g_teacher)**2)
                # jax.debug.print("{x}", x=jnp.mean(rewards))

                return h_loss, g_loss, jnp.mean(rewards)

            def h_loss_fn(h_params, rng, eps):
                h_loss, _, _ = loss_fn(h_params, t_state_g.params, rng, eps)
                return h_loss

            def g_loss_fn(g_params, rng, eps):
                _, g_loss, _ = loss_fn(t_state_h.params, g_params, rng, eps)
                return g_loss

            def get_losses(rng, eps):
                h_loss, g_loss, _ = loss_fn(t_state_h.params, t_state_g.params, rng, eps)
                return h_loss, g_loss
    
            rng, eps = x
            t_state_h, t_state_g = carry
            grad_h_loss_fn = jax.grad(h_loss_fn)
            grad_h = grad_h_loss_fn(t_state_h.params, rng, eps)
            t_state_h = t_state_h.apply_gradients(grads = grad_h)
            
            grad_g_loss_fn = jax.grad(g_loss_fn)
            grad_g = grad_g_loss_fn(t_state_g.params, rng, eps)
            t_state_g = t_state_g.apply_gradients(grads = grad_g)
            h_loss, g_loss = get_losses(rng, eps)
            return (t_state_h, t_state_g), (h_loss, g_loss)
        
        eps_v = jax.vmap(eps_policy, in_axes=(None, None, 0, 0))
        (t_state_h, t_state_g), (h_losses, g_losses) = jax.lax.scan(training_step, (t_state_h, t_state_g), (rngs, eps))

        return t_state_h, t_state_g, h_losses, g_losses

    # later vmapped
    def init_train_states(hinter, guesser, init_rng):
        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], is_dropout=True)
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], is_dropout=True)
        return t_state_h, t_state_g

    # declare static variables
    num_agents = config["num_agents"]
    batch_size = config["batch_size"]
    num_episodes = config["num_episodes"]
    N = config["N"]
    hg_env = HintGuessEnv(config)
    model = config["model"]
    train_rng = jax.random.PRNGKey(config["train_rng"])
    init_rng = jax.random.PRNGKey(config["init_rng"])
    eps_min = config["eps_min"]
    eps_max = config["eps_max"]
    K = config["K"]
    eval_interval = config["eval_interval"]
 
    # batched training setup
    # setup rngs and eps
    train_rngs = jax.random.split(train_rng, num_episodes * num_agents).reshape(num_agents, -1, 2)
    init_rngs = jax.random.split(init_rng, num_agents)
    n = jnp.arange(num_episodes)
    eps = eps_min + (eps_max - eps_min) * jnp.exp(-n/K)

   
    # init batched train_state
    init_sp, init_h1, init_h2, hinter_student, guesser_student = init_model(config)
    batch_init = jax.vmap(init_train_states, in_axes=(None, None, 0), out_axes=(0))
    batch_t_state_h, batch_t_state_g = batch_init(hinter_student, guesser_student, init_rngs) 

    # batched_train & jit
    batch_train = jax.vmap(train_sigle_agent, in_axes=(0, 0, 0, 0, 0, None,), out_axes=0)
    jitted_batch_train = jax.jit(batch_train)

    
    # train_agents
    hinter_teachers, guesser_teachers = teachers
    batch_t_state_h, batch_t_state_g, h_losses, g_losses = jitted_batch_train(train_rngs, batch_t_state_h, batch_t_state_g, hinter_teachers, guesser_teachers, eps)
            
    return batch_t_state_h, batch_t_state_g, h_losses, g_losses