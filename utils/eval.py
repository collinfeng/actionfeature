import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

from environments.hintguess import *
from utils.utils import *


def play_eval(t_state_h, t_state_g, rng, config):
    def greedy_policy(q_values):
        return jnp.argmax(q_values)

    def index_convertion(twohot_index):
        '''
        input: format of (feature_dim)
        output: format of (idx), idx <- {0, 8} if feature = 3, 1A -> 0, 3C ->8
        '''
        pos = jnp.argwhere(twohot_index==1, size=2)
        idx = (2 - pos[0])*3 + (2 - (pos[1] - 3))
        return idx
    
    greedy_v = jax.vmap(greedy_policy, in_axes=(0))
    index_convertion_v = jax.vmap(index_convertion)

    batch_size = config["batch_size"]
    N = config["N"]
    hg_env = HintGuessEnv(config)
    card_dim = 2 * config["feature_dim"]

    conditional_dim = config["feature_dim"]**2
    action_occurrence = jnp.zeros((conditional_dim, conditional_dim), jnp.int32) # use ones for safe division
    total_occurrence = jnp.zeros((conditional_dim, conditional_dim), jnp.int32)
    h_losses = []
    g_losses = []
    mean_rewards = []

    for eval_run in tqdm(range(config["eval_runs"])):
        rng, subrng = jax.random.split(rng)
        tgt_twohot, H1_twohot, H2_twohot = hg_env.get_observation(subrng)

        q_values_h = t_state_h.apply_fn({"params": t_state_h.params}, tgt_twohot, H2_twohot, H1_twohot)

        rng, subrng = jax.random.split(rng)
        rngs = jax.random.split(subrng, batch_size)
        h_actions = greedy_v(q_values_h)
        q_h = jnp.take_along_axis(q_values_h, h_actions[:, jnp.newaxis], axis=1).squeeze(axis=1)
        hinted_twohot = jnp.take_along_axis(H1_twohot, h_actions[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
        q_values_g = t_state_g.apply_fn({"params": t_state_g.params}, hinted_twohot, H1_twohot, H2_twohot)

        rng, subrng = jax.random.split(rngs[-1])
        rngs = jax.random.split(subrng, batch_size)
        guess = greedy_v(q_values_g)
        q_g = jnp.take_along_axis(q_values_g, guess[:, jnp.newaxis], axis=1).squeeze(axis=1)
        guess_twohot = jnp.take_along_axis(H2_twohot, guess[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
        rewards = hg_env.get_reward(tgt_twohot, guess_twohot)

        #counting occurence
        '''
        one hint correspond to N cards of a hand
        one hint correspond to one guess
        '''
        flatten_hint = hinted_twohot.reshape(-1, card_dim)
        hint_column_idx = index_convertion_v(flatten_hint)
        flatten_hand = H2_twohot.reshape(-1, card_dim)
        hand_row_idx = index_convertion_v(flatten_hand).flatten()
        total_occurrence = total_occurrence.at[hand_row_idx, jnp.repeat(hint_column_idx, N)].add(1)
        flatten_action = guess_twohot.reshape(-1, card_dim)
        guess_row_idx = index_convertion_v(flatten_action)
        action_occurrence = action_occurrence.at[guess_row_idx, hint_column_idx].add(1)
        # aux data logging
        h_loss = jnp.mean((q_h - rewards)**2)
        g_loss = jnp.mean((q_g - rewards)**2)
        h_losses.append(h_loss)
        g_losses.append(g_loss)
        mean_rewards.append(jnp.mean(rewards))
        # print(jnp.sum(action_occurrence), jnp.sum(total_occurrence))

    weigthed_occurrence = action_occurrence/total_occurrence
    # print(weigthed_occurrence)
    # jax.nn.softmax(weigthed_occurrence, axis=-1)
    # return jnp.array(h_losses).mean(), jnp.array(g_losses).mean(), jnp.array(mean_rewards).mean(), jax.nn.softmax(weigthed_occurrence, axis=-1)

    return jnp.array(mean_rewards).mean(), action_occurrence/total_occurrence

def xp_eval(agents, config):
    num_agents = config["num_agents"]
    xp_result = np.zeros((num_agents, num_agents))
    rng = jax.random.PRNGKey(config["eval_rng"])
    for i in range(num_agents):
        hinter_tx = agents[i][0]
        for j in range(num_agents):
            rng, subrng = jax.random.split(rng)
            guesser_tx = agents[j][1]
            reward, _ = play_eval(hinter_tx, guesser_tx, subrng, config)
            xp_result[i, j] = reward
            if config["xpeval_print"] == True:
                print(f"Hinter{i}, Guesser{j}, Reward: {reward}")

    return xp_result

def batched_xp_eval(batch_t_state_h, batch_t_state_g, config):
    # jitted later after vmapped
    def single_pair_eval(t_state_h, t_state_g):
        def greedy_policy(q_values):
            return jnp.argmax(q_values)

        def eval_step(t_state_h, t_state_g, rng):
            rng, subrng = jax.random.split(rng)
            tgt_twohot, H1_twohot, H2_twohot = hg_env.get_observation(subrng)

            q_values_h = t_state_h.apply_fn({"params": t_state_h.params}, tgt_twohot, H2_twohot, H1_twohot)

            rng, subrng = jax.random.split(rng)
            rngs = jax.random.split(subrng, batch_size)
            h_actions = greedy_v(q_values_h)
            hinted_twohot = jnp.take_along_axis(H1_twohot, h_actions[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
            q_values_g = t_state_g.apply_fn({"params": t_state_g.params}, hinted_twohot, H1_twohot, H2_twohot)

            rng, subrng = jax.random.split(rngs[-1])
            rngs = jax.random.split(subrng, batch_size)
            guess = greedy_v(q_values_g)
            guess_twohot = jnp.take_along_axis(H2_twohot, guess[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze(axis=1)
            rewards = hg_env.get_reward(tgt_twohot, guess_twohot)

            return rewards

        greedy_v = jax.vmap(greedy_policy, in_axes=(0))
        eval_step_v = jax.vmap(eval_step, in_axes=(None, None, 0))
        batch_size = config["batch_size"]
        hg_env = HintGuessEnv(config)
        eval_rng = jax.random.PRNGKey(config["eval_rng"])
        eval_rng = jax.random.split(eval_rng, config["batched_eval_runs"])
        rewards = eval_step_v(t_state_h, t_state_g, eval_rng)
        return rewards.mean()
    
    @jax.jit
    def tree_node_xp_op(batch_t_state_h, batch_t_state_g):
        def process_tree(x):
            dim = x.ndim
            shape = np.array(x.shape)
            shape[0] = shape[0] * config["num_agents"]
            axes = np.arange(dim - 1) + 1
            axes = np.concatenate((np.arange(x.ndim) + 1, [0]), axis=-1)
            x = jnp.repeat(x[jnp.newaxis, ...], config["num_agents"], axis=0)
            x = jnp.transpose(x, axes=axes)
            x = x.reshape(shape)
            return x
        def repeat(x):
            return jnp.repeat(x, config["num_agents"], axis=0)
        batch_t_state_h = jax.tree_map(process_tree, batch_t_state_h)
        batch_t_state_g = jax.tree_map(repeat, batch_t_state_g)
        return batch_t_state_h, batch_t_state_g
    
    batch_t_state_h, batch_t_state_g = tree_node_xp_op(batch_t_state_h, batch_t_state_g)
    vmap_xp_eval = jax.jit(jax.vmap(single_pair_eval))
    xp_scores = vmap_xp_eval(batch_t_state_h, batch_t_state_g)
    return xp_scores.mean()
