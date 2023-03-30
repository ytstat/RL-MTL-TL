#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:45:17 2023

@author: yetian
"""

# import numpy as np
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
from sklearn.linear_model import LinearRegression
from autograd.numpy import sqrt
from numpy.linalg import norm, svd
from numpy import log
from joblib import Parallel, delayed
import csv
import os


seed = int(os.getenv("SLURM_ARRAY_TASK_ID"))
np.random.seed(seed)

def MTL_ours(x, y, r = 3, T1 = 1, T2 = 0.05, R = 5, r_bar = 5, eta = 0.05, max_iter = 2000, C1 = 1, C2 = 0.5, delta = 0.05, adaptive = False):
    T = x.shape[0]
    n = x.shape[1]
    p = x.shape[2]
    

    ## adaptive or not
    if (adaptive == True):
        threshold = T1*sqrt((p+log(T))/n) + T2*R*(r_bar**(-3/4))
        # single-task linear regression
        beta_hat_single_task = np.zeros((p, T))
        for t in range(T):
            beta_hat_single_task[:, t] = LinearRegression().fit(x[t, :, :], y[t, :]).coef_
            length = norm(beta_hat_single_task[:, t])
            if (length > R):
                beta_hat_single_task[:, t] = beta_hat_single_task[:, t]/R
            
        r = max(np.where(svd(beta_hat_single_task/sqrt(T))[1] > threshold)[0])+1

    A_hat = np.zeros((T, p, r))
    A_bar = np.zeros((p, r), dtype='float64')
    A_bar[0:r, 0:r] = np.identity(r,dtype='float64')
    for t in range(T):
        A_hat[t, 0:r, 0:r] = np.identity(r)
        
    theta_hat = np.zeros((r, T))

    ## initialization
    for t in range(T):
        def ftotal(A, theta):
            return(1/n*np.dot(y[t, :] - x[t, :, :] @ A @ theta, y[t, :] - x[t, :, :] @ A @ theta))
            
        ftotal_grad = grad(ftotal, argnum = [0, 1])
        
        for i in range(200):
            S = ftotal_grad(A_hat[t, :, :], theta_hat[:, t])
            A_hat[t,:,:] = A_hat[t,:,:] - eta*S[0]
            theta_hat[:, t] = theta_hat[:, t] - eta*S[1]
      

    ## Step 1
    # lam = sqrt(r*(p+log(T)))*1
    lam = sqrt(r*(p+log(T)))*C1
    # lam1 = 2
    
    # loss = np.zeros(1000)
    for j in range(max_iter):
        def ftotal(A, theta, A_bar):
            s = 0
            for t in range(T):
                s = s + 1/n*1/T*np.dot(y[t, :] - x[t, :, :] @ A[t, :, :] @ theta[:, t], y[t, :] - x[t, :, :] @ A[t, :, :] @ theta[:, t]) + lam/sqrt(n)*1/T*max(abs(np.linalg.eigh(A[t, :, :] @ A[t, :, :].T - A_bar @ A_bar.T)[0]))
            s = s + delta*max(abs(np.linalg.eigh(A_bar.T @ A_bar - theta @ theta.T)[0]))
            return(s)
        
        ftotal_grad = grad(ftotal, argnum = [0,1,2])
        
        S = ftotal_grad(A_hat, theta_hat, A_bar)
        A_hat = A_hat- eta*S[0]
        theta_hat = theta_hat - eta*S[1]
        A_bar = A_bar - eta*S[2]

        beta_hat_step1 = np.zeros((p, T))

        
    beta_hat_step1 = np.zeros((p, T))
    for t in range(T):
        beta_hat_step1[:, t] = A_hat[t,:,:] @ theta_hat[:, t]
        
        
        
    # Step 2
    gamma = sqrt(p+log(T))*C2
    beta_hat_step2 = np.zeros((p, T))
    for t in range(T):
        def f(beta):
            return(1/n*np.dot(y[t, :] - x[t, :, :] @ beta, y[t, :] - x[t, :, :] @ beta) + gamma/sqrt(n)*(sum((beta - beta_hat_step1[:, t])**2))**0.5)
        f_grad = grad(f)
        for j in range(max_iter):
            S = f_grad(beta_hat_step2[:, t])
            beta_hat_step2[:, t] = beta_hat_step2[:, t] - eta*S
            
    return(beta_hat_step2)


def MTL(x, y, r, eta = 0.05, delta = 0.05, max_iter = 2000):
    T = x.shape[0]
    n = x.shape[1]
    p = x.shape[2]
    A_hat = np.zeros((p, r))
    
    for t in range(T):
        A_hat[0:r, 0:r] = np.identity(r)
        
    theta_hat = np.zeros((r, T))

    ## initialization
    t = 0
    def ftotal(A, theta):
        return(1/n*np.dot(y[t, :] - x[t, :, :] @ A @ theta, y[t, :] - x[t, :, :] @ A @ theta))
        
    ftotal_grad = grad(ftotal, argnum = [0, 1])
    
    for i in range(200):
        S = ftotal_grad(A_hat, theta_hat[:, t])
        A_hat = A_hat - eta*S[0]
        theta_hat[:, t] = theta_hat[:, t] - eta*S[1]
      

    ## Step 1
    
    # lam1 = 2
    for j in range(max_iter):
        def ftotal(A, theta):
            s = 0
            for t in range(T):
                s = s + 1/n*1/T*np.dot(y[t, :] - x[t, :, :] @ A @ theta[:, t], y[t, :] - x[t, :, :] @ A @ theta[:, t])
            s = s + delta*max(abs(np.linalg.eigh(A.T @ A - theta @ theta.T)[0]))
            return(s)
        
        ftotal_grad = grad(ftotal, argnum = [0,1])
        
        S = ftotal_grad(A_hat, theta_hat)
        A_hat = A_hat - eta*S[0]
        theta_hat = theta_hat - eta*S[1]

        
    beta_hat_step1 = np.zeros((p, T))
    for t in range(T):
        beta_hat_step1[:, t] = A_hat @ theta_hat[:, t]
        
    return(beta_hat_step1)

    
def avg_distance(beta_hat, beta):
    s = 0
    T = beta.shape[1]
    for t in range(T):
        s = s + norm(beta_hat[:, t]-beta[:, t])/T
    return(s)

def max_distance(beta_hat, beta):
    T = beta.shape[1]
    s = np.zeros(T)
    for t in range(T):
        s[t] = norm(beta_hat[:, t]-beta[:, t])
    return(max(s))

def all_distance(beta_hat, beta):
    T = beta.shape[1]
    s = np.zeros(T)
    for t in range(T):
        s[t] = norm(beta_hat[:, t]-beta[:, t])
    return(s)



def our_task(h):
    n = 100; p = 20; r = 3; T = 6

    ## parameter setting: 0 outlier
    theta = np.array([[1,0.5,0], [1,-1,1], [1.5,1.5,0], [1,1,0], [1,0,1], [-1,-1,-1]]).T*2
    R = np.random.normal(0, 1, p*p).reshape((p,p))
    A_center = (np.linalg.svd(R)[0])[0:r, ]
    A_center = A_center.T # p*r matrix
    A = np.zeros((T, p, r))
    beta = np.zeros((p, T))
    for t in range(T):
        Delta_A = np.zeros((p, r))
        Delta_A[0:r, 0:r] = np.random.uniform(low=-h,high=h,size=1)*np.identity(r)
        A[t, :, :] = A_center + Delta_A
        beta[:, t] = A[t, :, :] @ theta[:, t]
        
    ## data generation
    x = np.zeros((T, n, p))
    y = np.zeros((T, n))

    for t in range(T):
        x[t, :, :] = np.random.normal(0, 1, n*p).reshape((n,p))
        y[t, :] = x[t, :, :] @ beta[:, t] + np.random.normal(0, 1, n)


    # single-task linear regression
    beta_hat_single_task = np.zeros((p, T))
    for t in range(T):
        beta_hat_single_task[:, t] = LinearRegression().fit(x[t, :, :], y[t, :]).coef_


    # MTL ours
    beta_hat_ours = MTL_ours(x = x, y = y, r = 3, max_iter = 2000, eta = 0.05)
    
    # MTL ours: adaptive
    beta_hat_ours_ad = MTL_ours(x = x, y = y, r = 3, r_bar = sqrt(T), eta = 0.05, max_iter = 2000, adaptive = True)
   
    
    # MTL the same representation
    beta_hat = MTL(x = x, y = y, r = 3, eta = 0.05, max_iter = 2000)
    
    result = np.zeros(4)
    result[0] = max(all_distance(beta_hat_single_task, beta)[0:T])
    result[1] = max(all_distance(beta_hat, beta)[0:T])
    result[2] = max(all_distance(beta_hat_ours, beta)[0:T])
    result[3] = max(all_distance(beta_hat_ours_ad, beta)[0:T])
    return(result)



def our_task_outlier(h):
    n = 100; p = 20; r = 3; T = 6

    ## parameter setting: 1 outlier
    theta = np.array([[1,0.5,0], [1,-1,1], [1.5,1.5,0], [1,1,0], [1,0,1], [-1,-1,-1]]).T*2
    R = np.random.normal(0, 1, p*p).reshape((p,p))
    A_center = (np.linalg.svd(R)[0])[0:r, ]
    A_center = A_center.T # p*r matrix
    A = np.zeros((T, p, r))
    beta = np.zeros((p, T))
    for t in range(T):
        Delta_A = np.zeros((p, r))
        Delta_A[0:r, 0:r] = np.random.uniform(low=-h,high=h,size=1)*np.identity(r)
        A[t, :, :] = A_center + Delta_A
        beta[:, t] = A[t, :, :] @ theta[:, t]

    beta_outlier = np.random.uniform(-1,1,p)
    beta = np.hstack((beta, beta_outlier.reshape(p,1)))

    T = 7
        
    ## data generation
    x = np.zeros((T, n, p))
    y = np.zeros((T, n))

    for t in range(T):
        x[t, :, :] = np.random.normal(0, 1, n*p).reshape((n,p))
        y[t, :] = x[t, :, :] @ beta[:, t] + np.random.normal(0, 1, n)


    # single-task linear regression
    beta_hat_single_task = np.zeros((p, T))
    for t in range(T):
        beta_hat_single_task[:, t] = LinearRegression().fit(x[t, :, :], y[t, :]).coef_

    # MTL ours
    beta_hat_ours = MTL_ours(x = x, y = y, r = 3, max_iter = 2000, eta = 0.05)
    
    # MTL ours: adaptive
    beta_hat_ours_ad = MTL_ours(x = x, y = y, r = 3, r_bar = sqrt(T), max_iter = 2000, eta = 0.05, adaptive = True)
    
    
    # MTL the same representation
    beta_hat = MTL(x = x, y = y, r = 3, eta = 0.05, max_iter = 2000)
    
    result = np.zeros(8)
    result[0] = max(all_distance(beta_hat_single_task, beta)[0:(T-1)])
    result[1] = max(all_distance(beta_hat, beta)[0:(T-1)])
    result[2] = max(all_distance(beta_hat_ours, beta)[0:(T-1)])
    result[3] = max(all_distance(beta_hat_ours_ad, beta)[0:(T-1)])
    result[4] = all_distance(beta_hat_single_task, beta)[T-1]
    result[5] = all_distance(beta_hat, beta)[T-1]
    result[6] = all_distance(beta_hat_ours, beta)[T-1]
    result[7] = all_distance(beta_hat_ours_ad, beta)[T-1]
    
    return(result)






h_list = np.arange(0,0.9,0.1)

mse_no_outlier = np.zeros((h_list.size, 4))
mse_no_outlier = np.array(Parallel(n_jobs=4)(delayed(our_task)(h) for h in h_list))

mse_outlier = np.zeros((h_list.size, 8))
mse_outlier = np.array(Parallel(n_jobs=4)(delayed(our_task_outlier)(h) for h in h_list))


mse_no_outlier = mse_no_outlier.reshape((1, h_list.size*4))
mse_outlier = mse_outlier.reshape((1, h_list.size*8))

with open("/burg/home/yt2661/projects/RL-MTL/experiments/no_outlier/result/"+str(seed)+".csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(mse_no_outlier)
    
with open("/burg/home/yt2661/projects/RL-MTL/experiments/outlier/result/"+str(seed)+".csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(mse_outlier)
    

        