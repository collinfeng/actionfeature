import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
import orbax.checkpoint as ocp
from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot as plt
from models.hg_models import *
from utils.evaluations import *
from flax.training import train_state

class TrainState(train_state.TrainState):
  key: jax.Array

def stack_pytree_in_lst(lst_of_trees):

    def stack_op(first_node, remaining_nodes):
        batch_node = jnp.stack([first_node, remaining_nodes])
        return batch_node
    result = jax.tree_map(stack_op, lst_of_trees[0], lst_of_trees[1:])
    return result

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

def save_jax_array(ndarray, path="results", filename="untitled.npy"):
    with open(f"{path}/{filename}", 'wb') as f:
        jnp.save(f, ndarray)

def plot_sp_xp_result(suffix, config, save=False, agent=0, plot_xp=False):
    sp = np.load(f"results/{suffix}/sp_train_scores")
    plt.plot(sp[agent], label = "sp reward")
    if plot_xp:
        eval_interval = config["eval_interval"] 
        num_episodes = config["num_episodes"]
        num_evals = num_episodes // eval_interval
        scaled_iterations = [i * eval_interval for i in range(num_evals)]
        xp = np.load(f"results/{suffix}/xp_train_scores")
        plt.plot(scaled_iterations, xp, label = "xp reward")
    plt.legend() 
    if save:
        plt.savefig(f"results/{suffix}/sp-xp-result.png")
    plt.show()

def init_model(config):
    model = config["model"]
    init_sp = jnp.zeros((config["batch_size"], 2 * config["feature_dim"]), jnp.float32)
    init_h1 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)
    init_h2 = jnp.zeros((config["batch_size"], config["N"], 2 * config["feature_dim"]), jnp.float32)

    if config["model_type"] == "no_action":  
        hinter = model(hidden=config["mlp_hidden"],
                        batch_size=config["batch_size"],
                        N = config["N"])
        guesser = model(hidden=config["mlp_hidden"],
                        batch_size=config["batch_size"],
                        N = config["N"]) 
    elif config["model_type"] == "action_in":
        hinter = model(batch_size=config["batch_size"],
                        N=config["N"],
                        qkv_features=config["qkv_features"],
                        drop_out=config["dropout"])
                
        guesser = model(batch_size=config["batch_size"],
                        N=config["N"],
                        qkv_features=config["qkv_features"],
                        drop_out=config["dropout"])   
    else:
        raise ValueError('cannot init this model type')

    return init_sp, init_h1, init_h2, hinter, guesser

def create_train_state(model, init_sp, init_h1, init_h2, init_rng, lr, ckpt=None, is_dropout=False):
    optim = optax.adam(lr)
    if ckpt != None:
        optim.init(ckpt["opt_state"])
        return train_state.TrainState.create(apply_fn=model.apply, params=ckpt["params"], tx=optim)
    else:
        if not is_dropout:
            params = model.init({"params": init_rng}, init_sp, init_h1, init_h2)["params"]
            return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optim)
        else:
            params = model.init({"params": init_rng, 'dropout': init_rng}, init_sp, init_h1, init_h2, training=False)["params"]
            return TrainState.create(apply_fn=model.apply, params=params, key=init_rng, tx=optim)


def plot_cond_prob(suffix, config, save=False):
    init_sp, init_h1, init_h2, hinter, guesser = init_model(config)
    init_rng = jax.random.PRNGKey(config["init_rng"])
    labels = [f"{char}{num}" for char in "ABC" for num in range(1, 4)]
    fig, axs = plt.subplots(1, config["num_agents"], figsize=(60, 5))  

    for i, ax in enumerate(axs):
        guesser_idx = i
        hinter_idx = i  # Replace with how you determine hinter_idx if it's different from guesser_idx
        h_tree = load_trainstate(f"checkpoints/{suffix}/hinter_{hinter_idx}")
        g_tree = load_trainstate(f"checkpoints/{suffix}/guesser_{guesser_idx}")

        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], params=h_tree["params"], dropout_rng=init_rng)
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], params=g_tree["params"], dropout_rng=init_rng )
        
        rewards, conditional_prob = play_eval(t_state_h, t_state_g, init_rng, config)
        cax = ax.imshow(conditional_prob, cmap='Blues')  # Use the i-th conditional probability matrix
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f'Guesser #{guesser_idx}')
        ax.set_ylabel(f'Hinter #{hinter_idx}')
        # ax.set_title(f'Subplot {i}')  # Or you can put any title you want
        # print(rewards, conditional_prob)
    # Create an axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # This adds an axis for the colorbar
    fig.colorbar(cax, cax=cbar_ax)
    fig.suptitle(f'Conditional Probability of model {suffix}') 
    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect to not overlap with the colorbar
    if save:
        plt.savefig(f"results/{suffix}/cond_prob")
    plt.show()

def plot_xp(suffix, config, save=False):
    init_sp, init_h1, init_h2, hinter, guesser = init_model(config)
    init_rng = jax.random.PRNGKey(config["init_rng"])
    t_state_hs = []
    t_state_gs = []
    pairs = []
    for i in range(config["num_agents"]):
        h_tree = load_trainstate(f"checkpoints/{suffix}/hinter_{i}")
        g_tree = load_trainstate(f"checkpoints/{suffix}/guesser_{i}")
        t_state_h = create_train_state(hinter, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], params=h_tree["params"])
        t_state_g = create_train_state(guesser, init_sp, init_h1, init_h2, init_rng, config["learning_rate"], params=g_tree["params"] )
        pairs.append([t_state_h, t_state_g])
        # t_state_hs.append(t_state_h)
        # t_state_gs.append(t_state_g)

    # batch_t_state_h = stack_pytree_in_lst(t_state_hs)
    # batch_t_state_g = stack_pytree_in_lst(t_state_gs)
    # xp_result = batched_xp_eval_drop_out(batch_t_state_h, batch_t_state_g, config)

    xp_result = xp_eval(pairs, config)

    labels = [f"Agent {i}" for i in range(1, config["num_agents"]+1)]
    fig, ax = plt.subplots()
    cax = ax.imshow(xp_result, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar = fig.colorbar(cax, ax=ax)
    fig.suptitle(f'xp averaged: {xp_result.mean()}') 
    plt.tight_layout()
    if save:
        plt.savefig(f"results/{suffix}/xp_result.png")
    plt.show()


