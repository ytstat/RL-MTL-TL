#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relationship between task performance and signal strength in Section 5.1.4
"""

import numpy as np
import sys, os
import torch
import csv
from joblib import Parallel, delayed
from spicy.linalg import sqrtm

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


seed = task_id

print('We are running with seed ' + str(seed), flush=True)


# --------------------------------------
## Simulation settings
h = 0
epsilon = 0
n = 100; p = 50; r = 5; T = 10; 


file_path = '/moto/home/yt2661/work/RL-MTL/experiments/sim_theta/result/'+str(seed)+'.csv'

# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)


np.random.seed(seed)
torch.manual_seed(seed)

num_outlier = int(np.floor(epsilon*T)) 
S = np.sort(np.random.choice(range(T), size = T-num_outlier, replace = False))
Sc = np.setdiff1d(range(T), S)
theta_norm = np.arange(0.5, 5.5, 0.5)
theta = np.zeros((r, T))
for t in range(T):
    Z = np.random.normal(size = r)
    theta[:, t] = Z/np.linalg.norm(Z)*theta_norm[t]
    
R = np.random.normal(0, 1, p*p).reshape((p,p))
A_center = (np.linalg.svd(R)[0])[0:r, ]
A_center = A_center.T # p*r matrix


# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)


def run_sim():
    est_error_S = np.zeros((9, T))

    A = np.zeros((T, p, r))
    beta = np.zeros((p, T))

    # generate beta in S
    for t in S:
        Delta_A = np.zeros((p, r))
        Delta_A[0:r, 0:r] = np.random.uniform(low=-h,high=h,size=1)*np.identity(r)
        A[t, :, :] = A_center + Delta_A
        sqrt_ATA = sqrtm(A[t,:,:].T@A[t,:,:])
        sqrt_ATA[abs(sqrt_ATA)<=1e-10] = 0
        A[t, :, :] = A[t, :, :] @ sqrt_ATA
        beta[:, t] = A[t, :, :] @ theta[:, t]

    # generate beta outside S
    if num_outlier > 0:
        beta_outlier = np.random.uniform(low = -3, high = 3, size = num_outlier*p).reshape(p, num_outlier)
        # beta_outlier = np.random.uniform(low = -5, high = 5, size = num_outlier*p).reshape(p, num_outlier)
        beta[:, Sc] = beta_outlier
    

    # data generation
    train_data = []
    for t in range(T):
        if t in S:
            x = np.random.normal(0, 1, n*p).reshape((n,p))
        else:
            x = np.random.normal(0, 2, n*p).reshape((n,p))
        y = x @ beta[:, t] + np.random.normal(0, 1, n)
        train_data.append((x, y))
    
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
    
    # GLasso
    beta_hat_GLasso = GLasso(train_data)
    
    ## write the estimation error
    est_error_S[0, :] = all_distance(beta_hat_single_task, beta)[S]
    est_error_S[1, :] = all_distance(beta_hat_pooled, beta)[S]
    est_error_S[2, :] = all_distance(beta_hat_ERM, beta)[S]
    est_error_S[3, :] = all_distance(beta_hat_MoM, beta)[S]
    est_error_S[4, :] = all_distance(beta_hat_ARMUL, beta)[S]
    est_error_S[5, :] = all_distance(beta_hat_AdaptRep, beta)[S]
    est_error_S[6, :] = all_distance(beta_hat_pERM['step2'], beta)[S]
    est_error_S[7, :] = all_distance(beta_hat_spectral['step2'], beta)[S]
    est_error_S[8, :] = all_distance(beta_hat_GLasso, beta)[S]
    
    return(est_error_S)
    

est_error = run_sim() 


with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(est_error)