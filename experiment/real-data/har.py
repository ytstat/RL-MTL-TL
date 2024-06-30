#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-data analysis in Section 5.2
"""


import numpy as np
import sys, os
import pickle
import torch
import csv
import argparse


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


from mtl_func_torch import pERM, ERM, spectral, single_task_LR, pooled_LR
from funcs import prediction, all_classification_error
from ARMUL import ARMUL_blackbox

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
from grouplasso_supp import GLasso



task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
seed = task_id

print('We are running with seed ' + str(seed), flush=True)

np.random.seed(seed)
torch.manual_seed(seed)





# read the pre-processed data
with open('/burg/home/yt2661/projects/RL-MTL/datasets/har_standardized.pkl', 'rb') as f:
    (x_pca, y, subject) = pickle.load(f)
    

# assign data to each task according to subject value, split data into training/test sets
train_data = []
test_data = []
test_ratio = 0.5
for t in np.unique(subject):
    index = np.where(subject == t)[0]
    test_index = np.random.choice(index, size = int(np.floor(index.size*test_ratio)), replace = False)
    train_index = np.setdiff1d(index, test_index)
    
    train_data.append((x_pca[train_index,:], y[train_index]))
    test_data.append((x_pca[test_index,:], y[test_index]))


# read r value from the input in terminal: let r = 5, 10, 15
parser = argparse.ArgumentParser(description="apply a specified r value.")
parser.add_argument('r', type=int, help="r value")
args = parser.parse_args()
r = args.r
print(f"use r value: {r}", flush = True)



file_path = '/burg/home/yt2661/projects/RL-MTL/experiments/har/result/r_'+str(r) + '_'+str(seed)+'.csv'

# Check if the output file exists
if os.path.exists(file_path):
    print(f"File {file_path} exists. Stopping the script.", flush=True)
    sys.exit(0)
else:
    print(f"File {file_path} does not exist. Continuing the script.", flush=True)



def run_sim():
    est_error_S = np.zeros((7, 30))
    
    # pERM
    print('start running pERM!', flush = True)
    beta_hat_pERM = pERM(data = train_data, r = r, C1 = 1, C2 = 0.5, adaptive=False, link = 'logistic', max_iter=2000, info = False)
    y_pred_pERM = prediction(beta_hat_pERM['step2'], test_data)
    print('pERM ends!', flush = True)
    
    
    # ARMUL
    beta_hat_ARMUL = ARMUL_blackbox(train_data, r, eta = 0.1, L = 10, n_fold = 5, seed = seed, link = 'logistic', c_max = 5)
    y_pred_ARMUL = prediction(beta_hat_ARMUL, test_data)
    
    
    # single-task linear regression
    beta_hat_single_task = single_task_LR(train_data, link = 'logistic')
    y_pred_single_task = prediction(beta_hat_single_task, test_data)
    
    
    
    # pooled linear regression
    beta_hat_pooled = pooled_LR(train_data, link = 'logistic')
    y_pred_pooled = prediction(beta_hat_pooled, test_data)
    
    
    
    # ERM
    beta_hat_ERM = ERM(data = train_data, r = r, info = False, link = 'logistic')
    y_pred_ERM = prediction(beta_hat_ERM, test_data)
    
    
    
    # spectral
    beta_hat_spectral = spectral(data = train_data, r = r, C2 = 0.5, T1 = 0.25, T2 = 0.5, adaptive=False, link = 'logistic', info = True, q = 0.05)
    y_pred_spectral = prediction(beta_hat_spectral['step2'], test_data)
    
    
    # GLasso
    # transformation of y from 0, 1 to -1, +1 in training data
    for t in range(len(train_data)):
        train_data[t][1][train_data[t][1] == 0] = -1
        
    beta_hat_GLasso = GLasso(train_data)
    y_pred_GLasso = prediction(beta_hat_GLasso, test_data)
    
    
    ## write the estimation error
    est_error_S[0, :] = all_classification_error(y_pred_single_task, test_data)
    est_error_S[1, :] = all_classification_error(y_pred_pooled, test_data)
    est_error_S[2, :] = all_classification_error(y_pred_ERM, test_data)
    est_error_S[3, :] = all_classification_error(y_pred_ARMUL, test_data)
    est_error_S[4, :] = all_classification_error(y_pred_pERM, test_data)
    est_error_S[5, :] = all_classification_error(y_pred_spectral, test_data)
    est_error_S[6, :] = all_classification_error(y_pred_GLasso, test_data)
    
    return(est_error_S)


est_error = run_sim() 


with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(est_error)