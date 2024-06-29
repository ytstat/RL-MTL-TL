#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import sys, os
import torch
import csv
from joblib import Parallel, delayed

path1 = "/moto/home/yt2661/work/RL-MTL/code"
path2 = "/moto/home/yt2661/work/RL-MTL/code/benchmarks/ARMUL/"
path3 = "/moto/home/yt2661/work/RL-MTL/code/benchmarks/AdaptRep/"
path4 = "/moto/home/yt2661/work/RL-MTL/code/benchmarks/group-lasso/"
sys.path.append(os.path.join(os.path.dirname(path1)))
sys.path.append(os.path.join(os.path.dirname(path2)))
sys.path.append(os.path.join(os.path.dirname(path3)))
sys.path.append(os.path.join(os.path.dirname(path4)))


from mtl_func_torch import pERM, ERM, spectral, MoM, single_task_LR, pooled_LR
from funcs import all_distance
from ARMUL import ARMUL_blackbox
from AdaptRep import AdaptRep
from generate_data import generate_data


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
h_array = np.arange(0, 0.9, 0.1)
epsilon = 0

file_path = '/moto/home/yt2661/work/RL-MTL/experiments/sim_h/result/setting_'+str(setting_no)+'_'+str(seed)+'.csv'

# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)



def run_sim(h, setting_no):
    print('Running simulations with different h values... setting ' + str(setting_no) + '... h = ' + str(h), flush=True)
    est_error_S = np.zeros(8)
    
    if setting_no == 1:
        n = 100; p = 30; r = 5; T = 50; 
        
    if setting_no == 2:
        n = 100; p = 50; r = 5; T = 50; 
    
    if setting_no == 3:
        n = 100; p = 80; r = 5; T = 50; 
    
    if setting_no == 4:
        n = 150; p = 80; r = 10; T = 50; 

    output_dict = generate_data(n, p, r, T, h, epsilon)
    train_data = output_dict['data']
    beta = output_dict['beta']
    S = output_dict['S']
    
    ## run different methods
    # ARMUL
    beta_hat_ARMUL = ARMUL_blackbox(train_data, r, eta = 0.02, L = 10, n_fold = 5, seed = seed)
    
    # AdaptRep
    beta_hat_AdaptRep = AdaptRep(train_data, r, num_batches=500, batch_size = 64)
    
    # single-task linear regression
    beta_hat_single_task = single_task_LR(train_data)
    
    # pooled linear regression
    beta_hat_pooled = pooled_LR(train_data)

    # MoM
    beta_hat_MoM = MoM(train_data, r = r)
    
    # pERM
    beta_hat_pERM = pERM(data = train_data, r = r, C1 = 1, C2 = 1, info = False, adaptive=False)
    
    # ERM
    beta_hat_ERM = ERM(data = train_data, r = r, info = False)
    
    # spectral
    beta_hat_spectral = spectral(data = train_data, r = r, C2 = 0.5, info = False, adaptive=False)
    
    
    ## write the estimation error
    est_error_S[0] = max(all_distance(beta_hat_single_task, beta)[S])
    est_error_S[1] = max(all_distance(beta_hat_pooled, beta)[S])
    est_error_S[2] = max(all_distance(beta_hat_ERM, beta)[S])
    est_error_S[3] = max(all_distance(beta_hat_MoM, beta)[S])
    est_error_S[4] = max(all_distance(beta_hat_ARMUL, beta)[S])
    est_error_S[5] = max(all_distance(beta_hat_AdaptRep, beta)[S])
    est_error_S[6] = max(all_distance(beta_hat_pERM['step2'], beta)[S])
    est_error_S[7] = max(all_distance(beta_hat_spectral['step2'], beta)[S])
    
    return(est_error_S)
    

est_error = np.array(Parallel(n_jobs=n_cpu)(delayed(run_sim)(h, setting_no) for h in h_array))

est_error = est_error.T


with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(est_error)