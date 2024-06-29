#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptivity to the intrinsic dimension r in Section 5.1.5
"""

import numpy as np
import sys, os
import torch
import csv
from joblib import Parallel, delayed

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


from mtl_func_torch import pERM, spectral, select_r, single_task_LR
from funcs import all_distance
from generate_data import generate_data


# set the random seed
task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))



setting_no = task_id // 100 + 1
seed = task_id % 100

print('We are running setting ' + str(setting_no) + ' with seed ' + str(seed), flush=True)

np.random.seed(seed)
torch.manual_seed(seed)

# --------------------------------------
## Simulation settings
h_array = np.arange(0, 0.9, 0.1)
epsilon = 0

file_path = '/moto/home/yt2661/work/RL-MTL/experiments/sim_r_adaptive/result/setting_'+str(setting_no)+'_'+str(seed)+'.csv'

# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)



def run_sim(h, setting_no):
    print('Running simulations with different h values... setting ' + str(setting_no) + '... h = ' + str(h), flush=True)
    est_error_S = np.zeros(6)
     
    if setting_no == 1:
        n = 100; p = 50; r = 5; T = 50; epsilon = 0
        
    if setting_no == 2:
        n = 100; p = 50; r = 5; T = 50; epsilon = 0.04
    
    if setting_no == 3:
        n = 150; p = 80; r = 10; T = 50; epsilon = 0
    
    if setting_no == 4:
        n = 150; p = 80; r = 10; T = 50; epsilon = 0.04
        
    output_dict = generate_data(n, p, r, T, h, epsilon)
    train_data = output_dict['data']
    beta = output_dict['beta']
    S = output_dict['S']

    
    ## run different methods
    # estimate r
    r_hat = select_r(train_data, T1 = 0.5, T2 = 0.25, q = epsilon, epsilon_bar = epsilon)
    
    # single-task linear regression
    beta_hat_single_task = single_task_LR(train_data)
    
    
    # pERM
    beta_hat_pERM = pERM(data = train_data, r = r, C1 = 1, C2 = 1)
    beta_hat_pERM_adaptive = pERM(data = train_data, r = r_hat, C1 = 1, C2 = 1)

    
    # spectral
    beta_hat_spectral = spectral(data = train_data, r = r, C2 = 0.5)
    beta_hat_spectral_adaptive = spectral(data = train_data, r = r_hat, C2 = 0.5)

    
    ## write the estimation error
    est_error_S[0] = max(all_distance(beta_hat_pERM['step2'], beta)[S])
    est_error_S[1] = max(all_distance(beta_hat_pERM_adaptive['step2'], beta)[S])
    est_error_S[2] = max(all_distance(beta_hat_spectral['step2'], beta)[S])
    est_error_S[3] = max(all_distance(beta_hat_spectral_adaptive['step2'], beta)[S])
    est_error_S[4] = max(all_distance(beta_hat_single_task, beta)[S])
    est_error_S[5] = r_hat
    
    return(est_error_S)
    

est_error = np.array(Parallel(n_jobs=-1)(delayed(run_sim)(h, setting_no) for h in h_array))

est_error = est_error.T


with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(est_error)