from utils.train import * 
from utils.utils import *
from utils.evaluations import *
from utils.teach import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
import json


teacher_config = {
    "model": MLPModel,
    "train_func": train_agents,
    "model_type": "no_action", # action_in or no_action
    "N": 5,
    "feature_dim": 3, # this the number of classes under different features set, e.g. dim = 3 for 0, 1, 2
    "mlp_hidden": 128,
    "batch_size": 500,
    "learning_rate": 0.0001,
    "num_agents": 6,
    "init_rng": 123,
    }

attn3_config = {
    "model": AttnModel3,
    "model_type": "action_in", # action_in or no_action
    "N": 5,
    "feature_dim": 3, # this the number of classes under different features set, e.g. dim = 3 for 0, 1, 2
    "qkv_features":6,
    "num_episodes": 500000,
    "batch_size": 500,
    "learning_rate": 0.0001,
    "num_agents": 6,
    "init_rng": 123,
    "train_rng": 456,
    "dropout_rng": 789,
    "eval_rng": 432,
    "num_heads": 1,
    "eval_runs":20,
    "batched_eval_runs":10,
    "eps_min":0.01,
    "eps_max":0.95,
    "K":25000,
    "eval_interval":100000,
    "save_result": True,
    "dropout":0.1
    }

def init_train_states(hinter_teacher, guesser_teacher, init_rng, hinter_teacher_cp, guesser_teacher_cp):
	t_state_h = create_train_state(hinter_teacher, init_sp, init_h1, init_h2, init_rng, teacher_config["learning_rate"], ckpt=hinter_teacher_cp, is_dropout=False)
	t_state_g = create_train_state(guesser_teacher, init_sp, init_h1, init_h2, init_rng, teacher_config["learning_rate"], ckpt=guesser_teacher_cp, is_dropout=False)
	return t_state_h, t_state_g

teacher_loading_path = "checkpoints/2023-11-29/mlp"
batched_hinter_teacher_cp = load_trainstate(f"{teacher_loading_path}/batch_hinter")
batched_guesser_teacher_cp = load_trainstate(f"{teacher_loading_path}/batch_guesser")
init_sp, init_h1, init_h2, hinter_teacher, guesser_teacher = init_model(teacher_config)


init_rngs = jax.random.split(jax.random.PRNGKey(teacher_config["init_rng"]), teacher_config["num_agents"])
batched_tx_state_init = jax.vmap(init_train_states, in_axes=(None, None, 0, 0, 0), out_axes=(0, 0))
batched_teacher_hinter, batched_teacher_guesser = batched_tx_state_init(hinter_teacher, guesser_teacher, init_rngs, batched_hinter_teacher_cp, batched_guesser_teacher_cp)

teachers = (batched_teacher_hinter, batched_teacher_guesser)
batch_t_state_h, batch_t_state_g, sp_train_scores = teach_agents(attn3_config, teachers)

model_name = "attn3-teach"
currentDate = datetime.now().strftime("%Y-%m-%d")

if attn3_config["save_result"] == True:
	save_pytree(batch_t_state_h, f"checkpoints/{currentDate}/{model_name}/batch_hinter")
	save_pytree(batch_t_state_g, f"checkpoints/{currentDate}/{model_name}/batch_guesser")
	save_batched_pytree(batch_t_state_h, f"checkpoints/{currentDate}/{model_name}/hinter", attn3_config["num_agents"])
	save_batched_pytree(batch_t_state_g, f"checkpoints/{currentDate}/{model_name}/guesser", attn3_config["num_agents"])
	if not os.path.isdir("results/{currentDate}/{model_name}"):
		os.makedirs(f"results/{currentDate}/{model_name}")
	save_jax_array(sp_train_scores, f"results/{currentDate}/{model_name}", "sp_train_scores")


plot_sp_xp_result(f"{currentDate}/{model_name}", attn3_config, save=True, agent=0)
plot_cond_prob(f"{currentDate}/{model_name}", attn3_config, save=attn3_config["save_result"])
plot_xp(f"{currentDate}/{model_name}", attn3_config, save=attn3_config["save_result"])