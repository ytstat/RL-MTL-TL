#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational time with different number of tasks T in Section 5.1.3
"""

import numpy as np
import sys, os
import torch
import csv
from joblib import Parallel, delayed
import time

# replace the root path of the folder with your own
root_path = "/Users/yetian/Library/CloudStorage/Dropbox/Columbia/Research/Project/Representation-MTL/Code/public version/"

path1 = root_path
path2 = root_path + "benchmarks/ARMUL"
path3 = root_path + "benchmarks/AdaptRep"
path4 = root_path + "benchmarks/GLasso"
sys.path.append(os.path.join(os.path.dirname(path1)))
sys.path.append(os.path.join(os.path.dirname(path2)))
sys.path.append(os.path.join(os.path.dirname(path3)))
sys.path.append(os.path.join(os.path.dirname(path4)))


from mtl_func_torch import pERM, ERM, spectral, MoM, single_task_LR, pooled_LR
from funcs import all_distance
from ARMUL import ARMUL_blackbox
from AdaptRep import AdaptRep
from generate_data import generate_data

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
from grouplasso_supp import GLasso


# set the random seed
task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
n_cpu = int(os.getenv('SLURM_CPUS_PER_TASK'))

print('n_cpu = ' + str(n_cpu), flush = True)


setting_no = task_id // 100 + 1
seed = task_id % 100

print('We are running setting ' + str(setting_no) + ' with seed ' + str(seed), flush=True)

np.random.seed(seed)
torch.manual_seed(seed)

# --------------------------------------
## Simulation settings
T_array = np.array(range(10, 200, 15))


file_path = '/moto/home/yt2661/work/RL-MTL/experiments/sim_T_time/result/'+str(seed)+'.csv'

# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)



def run_sim(T):
    print('Running simulations with different T values... T = ' + str(T), flush=True)
    time_array = np.zeros(9)
    
    n = 100; p = 50; r = 5; epsilon = 0; h = 0

    output_dict = generate_data(n, p, r, T, h, epsilon)
    train_data = output_dict['data']
    beta = output_dict['beta']
    S = output_dict['S']
    Sc = np.setdiff1d(range(T), S)
    
    ## run different methods
    # ARMUL
    t1 = time.time()
    beta_hat_ARMUL = ARMUL_blackbox(train_data, r, eta = 0.02, L = 10, n_fold = 5, seed = seed)
    t2 = time.time()
    time_array[0] = t2 - t1
    
    # AdaptRep
    t1 = time.time()
    beta_hat_AdaptRep = AdaptRep(train_data, r, num_batches=500, batch_size = 64)
    t2 = time.time()
    time_array[1] = t2 - t1
    
    # single-task linear regression
    t1 = time.time()
    beta_hat_single_task = single_task_LR(train_data)
    t2 = time.time()
    time_array[2] = t2 - t1
    
    # pooled linear regression
    t1 = time.time()
    beta_hat_pooled = pooled_LR(train_data)
    t2 = time.time()
    time_array[3] = t2 - t1
    
    # MoM
    t1 = time.time()
    beta_hat_MoM = MoM(train_data, r = r)
    t2 = time.time()
    time_array[4] = t2 - t1
    
    # pERM
    t1 = time.time()
    beta_hat_pERM = pERM(data = train_data, r = r, C1 = 1, C2 = 1, info = False, adaptive=False)
    t2 = time.time()
    time_array[5] = t2 - t1
    
    # ERM
    t1 = time.time()
    beta_hat_ERM = ERM(data = train_data, r = r, info = False)
    t2 = time.time()
    time_array[6] = t2 - t1
    
    # spectral
    t1 = time.time()
    beta_hat_spectral = spectral(data = train_data, r = r, C2 = 0.5, info = False, adaptive=False, q = epsilon)
    t2 = time.time()
    time_array[7] = t2 - t1
    
    ## GLasso
    t1 = time.time()
    beta_hat_GLasso = GLasso(train_data)
    t2 = time.time()
    time_array[8] = t2 - t1
    
    return(time_array)
    

time_array = np.array(Parallel(n_jobs=n_cpu)(delayed(run_sim)(T) for T in T_array))


with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(time_array)
    