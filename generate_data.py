#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generation function for simulations
"""

import numpy as np
from scipy.linalg import sqrtm


# --------------------------------------
def generate_data(n = 100, p = 50, r = 5, T = 50, h = 0, epsilon = 0, link = 'linear', H = 2):
    num_outlier = int(np.floor(epsilon*T)) 
    S = np.sort(np.random.choice(range(T), size = T-num_outlier, replace = False))
    Sc = np.setdiff1d(range(T), S)
    theta = np.random.uniform(low = -H, high = H, size = T*r).reshape(r, T)
    R = np.random.normal(0, 1, p*p).reshape((p,p))
    A_center = (np.linalg.svd(R)[0])[0:r, ]
    A_center = A_center.T # p*r matrix
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
        beta[:, Sc] = beta_outlier

    # data generation
    train_data = []
    for t in range(T):
        if t in S:
            x = np.random.normal(0, 1, n*p).reshape((n,p))
        else:
            x = np.random.normal(0, 2, n*p).reshape((n,p))
        
        if link == 'linear':
            y = x @ beta[:, t] + np.random.normal(0, 1, n)
        elif link == 'logistic':
            prob = 1/(1+np.exp(-x @ beta[:, t]))
            y = np.random.binomial(1, prob)
            
        train_data.append((x, y))
    
    output_dict = {'data': train_data, 
                   'beta': beta, 
                   'S': S}
    return(output_dict)

