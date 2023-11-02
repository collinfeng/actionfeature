from utils.train import * 
from utils.utils import *
from utils.eval import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


if __name__ == "__main__":
    currentDate = datetime.now().strftime("%Y-%m-%d")
    currentTime = datetime.now().strftime("%H:%M:%S")
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
    "PRNGkey": 0,
    "eval_PRNGkey":12345,
    "num_heads": 1,
    "eval_runs":10,
    "eps_min":0.01,
    "eps_max":0.95,
    "K":50000,
    "logging":True
    }
    cp_suffix = "SA2I-QKV-C"
    train_agents(config, cp_suffix)

    



	
	
	
	
	
