from utils.train import * 
from utils.utils import *
from utils.evaluations import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
import json

if __name__ == "__main__":
    attn3_config = {
    "model": AttnModel3,
    "train_func": train_agents_dropout,
    "model_type": "action_in", # action_in or no_action
    "N": 5,
    "feature_dim": 3, # this the number of classes under different features set, e.g. dim = 3 for 0, 1, 2
    "qkv_features":6,
    "num_episodes": 100000,
    "batch_size": 500,
    "learning_rate": 0.0001,
    "num_agents": 12,
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
    "eval_interval":100,
    "save_result": True,
    "dropout":0.1
    }

    mlp_config = {
    "model": MLPModel,
    "train_func": train_agents,
    "model_type": "no_action", # action_in or no_action
    "N": 5,
    "feature_dim": 3, # this the number of classes under different features set, e.g. dim = 3 for 0, 1, 2
    "num_episodes": 500000,
    "mlp_hidden": 128,
    "batch_size": 500,
    "learning_rate": 0.0001,
    "num_agents": 6,
    "init_rng": 123,
    "train_rng": 456,
    "eval_rng": 432,
    "num_heads": 1,
    "eval_runs":20,
    "batched_eval_runs":10,
    "eps_min":0.01,
    "eps_max":0.95,
    "K":25000,
    "eval_interval":500000,
    "save_result": True,
    }

    config = attn3_config
    train_func = config["train_func"]
    batch_t_state_h, batch_t_state_g, sp_train_scores = train_func(config)
    model_name = "attn3"
    currentDate = datetime.now().strftime("%Y-%m-%d")
    
    if config["save_result"] == True:
        save_pytree(batch_t_state_h, f"checkpoints/{currentDate}/{model_name}/batch_hinter")
        save_pytree(batch_t_state_g, f"checkpoints/{currentDate}/{model_name}/batch_guesser")
        save_batched_pytree(batch_t_state_h, f"checkpoints/{currentDate}/{model_name}/hinter", config["num_agents"])
        save_batched_pytree(batch_t_state_g, f"checkpoints/{currentDate}/{model_name}/guesser", config["num_agents"])
        if not os.path.isdir("results/{currentDate}/{model_name}"):
            os.makedirs(f"results/{currentDate}/{model_name}")
        save_jax_array(sp_train_scores, f"results/{currentDate}/{model_name}", "sp_train_scores")
        # save_jax_array(xp_train_scores, f"results/{currentDate}/{model_name}", "xp_train_scores")
    
    # plot_sp_xp_result(f"{currentDate}/{model_name}", config, save=True, agent=0)
    # plot_cond_prob(f"{currentDate}/{model_name}", config, save=config["save_result"])
    # plot_xp(f"{currentDate}/{model_name}", config, save=config["save_result"])
    # with open(f'results/{currentDate}/{model_name}/config.json', 'w') as fp:
    #     fp.write(json.dump(config, fp))
    



	
	
	
	
	
