from utils.train import * 
from utils.general import *
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
    "emb_dim":6,
    "qkv_features":6,
    "out_features":6,
    "num_episodes": 4000000,
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
     
	train_agents(config)
	np.load("result/2023-10-31")
	
	
	
	
	
