from utils.train import * 
from utils.utils import *
from utils.evaluations import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


if __name__ == "__main__":
    config = {
	"model": SA2I2MLP,
    "N": 5,
    "feature_dim": 3, # this the number of classes under different features set, e.g. dim = 3 for 0, 1, 2
    "emb_dim":8,
    "qkv_features":8,
    "out_features":8,
    "num_episodes": 1000000,
    "mlp_hidden": 128,
    "batch_size": 500,
    "learning_rate": 0.0001,
    "num_agents": 6,
    "init_rng": 123,
    "train_rng": 456,
    "eval_rng": 789,
    "num_heads": 1,
    "eval_runs":10,
    "batched_eval_runs":10,
    "eps_min":0.01,
    "eps_max":0.95,
    "K":50000,
    "eval_interval":1000,
    "save_result": True,
    "xpeval_print": False,
    "training":True
    }


    cp_suffix = "SA2I-2MLP-Q"
    train_agents(config, cp_suffix)
    # model_test()
    

    



	
	
	
	
	
