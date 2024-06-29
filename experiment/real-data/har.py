#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:48:42 2024

@author: yetian
"""


import numpy as np
import sys, os
import pickle
import torch
import csv
from joblib import Parallel, delayed
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


path1 = "/moto/home/yt2661/work/RL-MTL/code/"
path2 = "/moto/home/yt2661/work/RL-MTL/code/benchmarks/ARMUL/"
path3 = "/moto/home/yt2661/work/RL-MTL/code/benchmarks/AdaptRep/"
path4 = "/moto/home/yt2661/work/RL-MTL/code/benchmarks/group-lasso/"
sys.path.append(os.path.join(os.path.dirname(path1)))
sys.path.append(os.path.join(os.path.dirname(path2)))
sys.path.append(os.path.join(os.path.dirname(path3)))
sys.path.append(os.path.join(os.path.dirname(path4)))


from mtl_func_torch import pERM, ERM, spectral, single_task_LR, pooled_LR
from funcs import all_distance, prediction, all_classification_error
from ARMUL import ARMUL_blackbox



task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
seed = task_id

print('We are running with seed ' + str(seed), flush=True)

np.random.seed(seed)
torch.manual_seed(seed)


file_path = '/moto/home/yt2661/work/RL-MTL/experiments/har/result/'+str(seed)+'.csv'

# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)



# read the pre-processed data
with open('/moto/home/yt2661/work/RL-MTL/datasets/har_pca.pkl', 'rb') as f:
    (x_pca, y, subject) = pickle.load(f)
    
# assign data to each task according to subject value, split data into training/test sets
train_data = []
test_data = []
test_ratio = 0.2

for t in np.unique(subject):
    index = np.where(subject == t)[0]
    test_index = np.random.choice(index, size = int(np.floor(index.size*test_ratio)), replace = False)
    train_index = np.setdiff1d(index, test_index)
    
    train_data.append((x_pca[train_index,:], y[train_index]))
    test_data.append((x_pca[test_index,:], y[test_index]))



r = 10
def run_sim():
    est_error_S = np.zeros(6)
     
    # pERM
    print('start running pERM!', flush = True)
    beta_hat_pERM = pERM(data = train_data, r = r, C1 = r**(3/4), C2 = 0.5, info = True, adaptive=False, link = 'logistic')
    y_pred_pERM = prediction(beta_hat_pERM['step2'], test_data)
    print('pERM ends!', flush = True)
    
    
    # ARMUL
    beta_hat_ARMUL = ARMUL_blackbox(train_data, r, eta = 0.02, L = 10, n_fold = 5, seed = seed, link = 'logistic')
    y_pred_ARMUL = prediction(beta_hat_ARMUL, test_data)
    
    
    # single-task linear regression
    beta_hat_single_task = single_task_LR(train_data, link = 'logistic')
    y_pred_single_task = prediction(beta_hat_single_task, test_data)
    
    
    
    # pooled linear regression
    beta_hat_pooled = pooled_LR(train_data, link = 'logistic')
    y_pred_pooled = prediction(beta_hat_pooled, test_data)
    
   
    # ERM
    beta_hat_ERM = ERM(data = train_data, r = r, info = True, link = 'logistic', delta = 0)
    y_pred_ERM = prediction(beta_hat_ERM, test_data)
    
    
    
    # spectral
    beta_hat_spectral = spectral(data = train_data, r = r, C2 = 0.5, T1 = 0.25, T2 = 0.5, adaptive=False, info = True, link = 'logistic')
    y_pred_spectral = prediction(beta_hat_spectral['step2'], test_data)
    
    
    ## write the estimation error
    est_error_S[0] = max(all_classification_error(y_pred_single_task, test_data))
    est_error_S[1] = max(all_classification_error(y_pred_pooled, test_data))
    est_error_S[2] = max(all_classification_error(y_pred_ERM, test_data))
    est_error_S[3] = max(all_classification_error(y_pred_ARMUL, test_data))
    est_error_S[4] = max(all_classification_error(y_pred_pERM, test_data))
    est_error_S[5] = max(all_classification_error(y_pred_spectral, test_data))
    
    return(est_error_S)


est_error = run_sim() 


with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(est_error)