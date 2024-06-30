#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of different MTL methods
"""

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression

def column_norm(A):
    norm_A = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        norm_A[j] = np.linalg.norm(A[:, j])
    return(norm_A)


## penalized ERM (Algorithm 1)
def pERM(data, r = 3, T1 = 1, T2 = 1, R = None, r_bar = None, lr = 0.01, max_iter = 2000, C1 = 1, C2 = 1, 
            delta = 0.05, adaptive = False, info = False, tol = 1e-6, link = 'linear'):
    if info:
        print("pERM starts running...", flush = True)
    
    T = len(data)
    n = np.array([x.shape[0] for (x,y) in data])
    p = data[0][0].shape[1]
    n_total = np.sum(n)
    
    ## initialization
    x = np.zeros((n_total, p))
    y = np.zeros(n_total)
    
    # calculate sample indices for each task
    task_range = []
    start_index = 0
    for t in range(T):
        task_range.append(range(start_index, start_index+n[t]))
        start_index += n[t]
    
    # stack the x and y arrays
    for t in range(T):
        x[task_range[t], :] = data[t][0]
        y[task_range[t]] = data[t][1]
    
    ## r adaptive or not
    if (adaptive == True):
        # single-task linear regression
        beta_hat_single_task = np.zeros((p, T))
        if link == 'linear':
            for t in range(T):
                beta_hat_single_task[:, t] = LinearRegression(fit_intercept = False).fit(x[task_range[t], :], y[task_range[t]]).coef_
        elif link == 'logistic':
            for t in range(T):
                beta_hat_single_task[:, t] = LogisticRegression(fit_intercept = False).fit(x[task_range[t], :], y[task_range[t]]).coef_

        norm_each_task = column_norm(beta_hat_single_task)
        if R is None:
            R = np.median(norm_each_task)*2
        
        for t in range(T):
            if (norm_each_task[t] > R):
                beta_hat_single_task[:, t] = beta_hat_single_task[:, t]/norm_each_task[t]*R
        
        # set up threshold
        if r_bar is None:
            r_bar = x.shape[1]
        threshold = T1*np.sqrt((p+np.log(T))/np.max(n)) + T2*R*(r_bar**(-3/4))
        r = max(np.where(np.linalg.svd(beta_hat_single_task/np.sqrt(T))[1] > threshold)[0])+1
        if info:
            print('Selected r = ' + str(r))

    # initialization
    y = torch.tensor(y, requires_grad=False)
    x = torch.tensor(x, requires_grad=False)
    A_hat = np.zeros((T, p, r), dtype ='float64')
    A_bar = np.zeros((p, r), dtype='float64')
    A_bar[0:r, 0:r] = np.identity(r,dtype='float64')
    
    for t in range(T):
        A_hat[t, 0:r, 0:r] = np.identity(r)

    theta_hat = np.zeros((r, T))
    
    # transform arrays to tensors
    A_hat = torch.tensor(A_hat, requires_grad=True)
    A_bar = torch.tensor(A_bar, requires_grad=True)
    theta_hat = torch.tensor(theta_hat, requires_grad=True)
    
    ## Step 1    
    lam = np.sqrt(r*(p+np.log(T)))*C1

    if link == 'linear':
        def ftotal(A, theta, A_bar):
            s = 0
            for t in range(T):
                s = s + 1/(2*n_total)*torch.dot(y[task_range[t]] - x[task_range[t], :] @ A[t, :, :] @ theta[:, t], y[task_range[t]] 
                                     - x[task_range[t], :] @ A[t, :, :] @ theta[:, t]) + lam*np.sqrt(n[t])/n_total*torch.linalg.svd(A[t, :, :] @ torch.linalg.inv(A[t, :, :].T @ A[t, :, :]) @ A[t, :, :].T - A_bar @ torch.linalg.inv(A_bar.T @ A_bar) @ A_bar.T)[1][0]
            return(s)
    elif link == 'logistic':
        def ftotal(A, theta, A_bar):
            s = 0
            for t in range(T):
                logits = torch.matmul(x[task_range[t], :], A[t, :, :] @ theta[:, t])
                s = s + 1/n_total*torch.dot(1-y[task_range[t]], logits) + 1/n_total*torch.sum(torch.log(1+torch.exp(-logits))) + lam*np.sqrt(n[t])/n_total*torch.linalg.svd(A[t, :, :] @ torch.linalg.inv(A[t, :, :].T @ A[t, :, :]) @ A[t, :, :].T - A_bar @ torch.linalg.inv(A_bar.T @ A_bar) @ A_bar.T)[1][0]

            return(s)


    optimizer = optim.Adam([A_hat, theta_hat, A_bar], lr=lr)
    
    loss_last = 1e8
    for i in range(max_iter):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute the loss (sum of the largest singular values)
        loss = ftotal(A_hat, theta_hat, A_bar)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Update the matrices
        optimizer.step()
        
        # Print the loss every 100 iterations
        if info:
            if (i + 1) % 100 == 0:
                print("Iteration {}/{}, Loss: {}".format(i+1, max_iter, loss.item()), flush = True)
        if abs(loss_last-loss.item())/loss.item() <= tol:
            if info:
                print("Already converged. Stopped early.", flush = True)
            break
        loss_last = loss.item()
    
    
    beta_hat_step1 = torch.zeros(p, T, dtype=torch.float64)
    for t in range(T):
        beta_hat_step1[:, t] = A_hat[t,:,:] @ theta_hat[:, t]
    
    beta_hat_step1 = beta_hat_step1.detach()
    if info:
        print("Step 1 is completed.\n", flush = True)
        
        
    ## Step 2   
    gamma = np.sqrt(p+np.log(T))*C2
    beta = torch.zeros(p, T, requires_grad = True, dtype=torch.float64)
    
    if link == 'linear':
        def ftotal2(beta):
            s = 0
            for t in range(T):
                s = s + 1/(2*n[t])*torch.dot(y[task_range[t]] - x[task_range[t], :] @ beta[:, t], y[task_range[t]] - x[task_range[t], :] @ beta[:, t]) + gamma/np.sqrt(n[t])*torch.norm(beta[:, t] - beta_hat_step1[:, t])
            return(s)
    elif link == 'logistic':
        def ftotal2(beta):
            s = 0
            for t in range(T):
                logits = torch.matmul(x[task_range[t], :], beta[:, t])
                s = s + 1/n[t]*torch.dot(1-y[task_range[t]], logits) + 1/n[t]*torch.sum(torch.log(1+torch.exp(-logits))) + gamma/np.sqrt(n[t])*torch.norm(beta[:, t] - beta_hat_step1[:, t])
            return(s)
                                                                                                                                          
    
    optimizer2 = optim.Adam([beta], lr=lr)
    loss_last = 1e8
    for i in range(max_iter):
        # Zero the gradients
        optimizer2.zero_grad()
        
        # Compute the loss (sum of the largest singular values)
        loss2 = ftotal2(beta)
        
        # Backward pass to compute gradients
        loss2.backward()
        
        # Update the matrices
        optimizer2.step()
        
        # Print the loss every 100 iterations
        if info:
            if (i + 1) % 100 == 0:
                print("Iteration {}/{}, Loss: {}".format(i+1, max_iter, loss2.item()), flush = True)
        if abs(loss_last-loss2.item())/loss2.item() <= tol:
            if info:
                print("Already converged. Stopped early.", flush = True)
            break
        loss_last = loss2.item()
    
    beta_hat_step1 = beta_hat_step1.numpy()
    beta_hat_step2 = beta.detach().numpy()
    if info:
        print("Step 2 is completed.\n", flush = True)
    
    if info:
        print("pERM stops running...", flush = True)
    
    return({"step1": beta_hat_step1, "step2": beta_hat_step2})
   
 
## ERM (the same representation across tasks)
def ERM(data, r, eta = 0.05, delta = 0.05, max_iter = 2000, lr = 0.01, info = False, tol = 1e-6, link = 'linear'):
    if info:
        print("ERM starts running...", flush = True)
    
    T = len(data)
    n = np.array([x.shape[0] for (x,y) in data])
    p = data[0][0].shape[1]
    n_total = np.sum(n)
    
    ## initialization
    x = np.zeros((n_total, p))
    y = np.zeros(n_total)
    
    # calculate sample indices for each task
    task_range = []
    start_index = 0
    for t in range(T):
        task_range.append(range(start_index, start_index+n[t]))
        start_index += n[t]
    
    # stack the x and y arrays
    for t in range(T):
        x[task_range[t], :] = data[t][0]
        y[task_range[t]] = data[t][1]
    
    
    y = torch.tensor(y, requires_grad=False)
    x = torch.tensor(x, requires_grad=False)
    A_hat = np.zeros((p, r), dtype ='float64')
    for t in range(T):
        A_hat[0:r, 0:r] = np.identity(r)
        
    # transform arrays to tensors
    A_hat = torch.tensor(A_hat, requires_grad=True)
    theta_hat = torch.zeros(r, T, requires_grad=True, dtype=torch.float64)
    
    if link == 'linear':
        def ftotal(A, theta):
            s = 0
            for t in range(T):
                s = s + 1/(2*n_total)*torch.dot(y[task_range[t]] - x[task_range[t], :] @ A @ theta[:, t], y[task_range[t]] - x[task_range[t], :] @ A @ theta[:, t])
            return(s)
    elif link == 'logistic':
        def ftotal(A, theta):
            s = 0
            for t in range(T):
                logits = torch.matmul(x[task_range[t], :], A @ theta[:, t])
                s = s + 1/n_total*torch.dot(1-y[task_range[t]], logits) + 1/n_total*torch.sum(torch.log(1+torch.exp(-logits)))

            return(s)

    optimizer = optim.Adam([A_hat, theta_hat], lr=lr)
    loss_last = 1e8
    for i in range(max_iter):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute the loss (sum of the largest singular values)
        loss = ftotal(A_hat, theta_hat)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Update the matrices
        optimizer.step()
        
        # Print the loss every 100 iterations
        if info:
            if (i + 1) % 100 == 0:
                print("Iteration {}/{}, Loss: {}".format(i+1, max_iter, loss.item()), flush = True)
        
        if abs(loss_last-loss.item())/loss.item() <= tol:
            if info:
                print("Already converged. Stopped early.", flush = True)
            break
        loss_last = loss.item()

    beta_hat = torch.zeros(p, T, dtype=torch.float64)
    for t in range(T):
        beta_hat[:, t] = A_hat @ theta_hat[:, t]
        
    beta_hat = beta_hat.detach().numpy()
    if info:
        print("ERM stops running...", flush = True)
        
    return(beta_hat)

## Single-task regression
def single_task_LR(data, link = 'linear'):
    T = len(data)
    p = data[0][0].shape[1]
    beta_hat = np.zeros((p, T))
    if link == 'linear':
        for t in range(T):
            beta_hat[:, t] = LinearRegression(fit_intercept = False).fit(data[t][0], data[t][1]).coef_
    elif link == 'logistic':
        for t in range(T):
            beta_hat[:, t] = LogisticRegression(fit_intercept = False).fit(data[t][0], data[t][1]).coef_
    return(beta_hat)


## Pooled regression
def pooled_LR(data, link = 'linear'):
    T = len(data)
    p = data[0][0].shape[1]
    x_all = np.empty((0, p))
    y_all = np.empty(0)
    for (x, y) in data:
        x_all = np.concatenate((x_all, x))
        y_all = np.concatenate((y_all, y))
    
    beta_hat = np.zeros((p, T))
    if link == 'linear':
        beta_fit = LinearRegression(fit_intercept = False).fit(x_all, y_all).coef_
    elif link == 'logistic':
        beta_fit = LogisticRegression(fit_intercept = False).fit(x_all, y_all).coef_
    
    for t in range(T):
        beta_hat[:, t] = beta_fit
    return(beta_hat)


## Estimation of r (Algorithm 3) 
def select_r(data, T1 = 0.5, T2 = 0.25, R = None, r_bar = None, q = 0.05, epsilon_bar = 0.05, link = 'linear'):
    n = np.array([x.shape[0] for (x,y) in data])
    T = len(data)
    p = data[0][0].shape[1]
    beta_hat_single_task = np.zeros((p, T))
    # var_est = np.zeros(T)
    if link == 'linear':
        for t in range(T):
            beta_hat_single_task[:, t] = LinearRegression(fit_intercept = False).fit(data[t][0], data[t][1]).coef_
    elif link == 'logistic':
        for t in range(T):
            beta_hat_single_task[:, t] = LogisticRegression(fit_intercept = False).fit(data[t][0], data[t][1]).coef_

    norm_each_task = column_norm(beta_hat_single_task)
    if R is None:
        R = np.quantile(norm_each_task, q)
    
    for t in range(T):
        if (norm_each_task[t] > R):
            beta_hat_single_task[:, t] = beta_hat_single_task[:, t]/norm_each_task[t]*R
    
    # set up threshold
    if r_bar is None:
        r_bar = p
    threshold = T1*np.sqrt((p+np.log(T))/np.max(n)) + T2*R*np.sqrt(epsilon_bar)
    sigval = np.linalg.svd(beta_hat_single_task/np.sqrt(T))[1]
    if len(np.where(sigval > threshold)[0]) > 0:
        r = max(np.where(sigval > threshold)[0])+1
        print('Threshold = ' + str(threshold) + ', selected r = ' + str(r))
        return(r)
    else:
        print('No r is selected. Too large threshold.')
        return(None)
    
    
## Spectral method (Algorithm 2)
def spectral(data, r, C2 = 1, T1 = 1, T2 = 1, R = None, r_bar = None, lr = 0.01, max_iter = 2000, info = False, adaptive = False, tol = 1e-6, link = 'linear', q = 0):
    if info:
        print("spectral starts running...")
    T = len(data)
    n = np.array([x.shape[0] for (x,y) in data])
    p = data[0][0].shape[1]
    n_total = np.sum(n)
    
    ## initialization
    x = np.zeros((n_total, p))
    y = np.zeros(n_total)
    
    # calculate sample indices for each task
    task_range = []
    start_index = 0
    for t in range(T):
        task_range.append(range(start_index, start_index+n[t]))
        start_index += n[t]
    
    # stack the x and y arrays
    for t in range(T):
        x[task_range[t], :] = data[t][0]
        y[task_range[t]] = data[t][1]
    
    ## r adaptive or not
    if (adaptive == True):
        # single-task linear regression
        beta_hat_single_task = np.zeros((p, T))
        if link == 'linear':
            for t in range(T):
                beta_hat_single_task[:, t] = LinearRegression(fit_intercept = False).fit(x[task_range[t], :], y[task_range[t]]).coef_
        elif link == 'logistic':
            for t in range(T):
                beta_hat_single_task[:, t] = LogisticRegression(fit_intercept = False).fit(x[task_range[t], :], y[task_range[t]]).coef_

        norm_each_task = column_norm(beta_hat_single_task)
        if R is None:
            R = np.median(norm_each_task)*2
        
        for t in range(T):
            if (norm_each_task[t] > R):
                beta_hat_single_task[:, t] = beta_hat_single_task[:, t]/norm_each_task[t]*R
        
        # set up threshold
        if r_bar is None:
            r_bar = x.shape[1]
        threshold = T1*np.sqrt((p+np.log(T))/np.max(n)) + T2*R*(r_bar**(-3/4))
        r = max(np.where(np.linalg.svd(beta_hat_single_task/np.sqrt(T))[1] > threshold)[0])+1
        if info:
            print('Threshold = ' + str(threshold) + ', selected r = ' + str(r))

    
    # single-task linear regression
    B = np.zeros((p, T))
    if link == 'linear':
        for t in range(T):
            B[:, t] = LinearRegression(fit_intercept = False).fit(x[task_range[t], :], y[task_range[t]]).coef_
    elif link == 'logistic':
        for t in range(T):
            B[:, t] = LogisticRegression(fit_intercept = False).fit(x[task_range[t], :], y[task_range[t]]).coef_
    
    
    # truncation for robustness
    norm_each_task = np.zeros(T)
    for t in range(T):
        norm_each_task[t] = np.linalg.norm(B[:, t])
    if q > 0:
        R = np.quantile(norm_each_task, 1-q)
        for t in range(T):
            B[:, t] = B[:, t]/norm_each_task[t]*R
    
    
    ## Step 1
    theta_hat = np.zeros((r, T))
    beta_hat_step1 = np.zeros((p, T))
    
    # calculate the central representation
    A_hat = np.linalg.svd(B)[0][:, 0:r]
    
    # calculate the theta estimate based on A_hat
    if link == 'linear':
        for t in range(T):
            theta_hat[:, t] = LinearRegression(fit_intercept = False).fit(x[task_range[t], :] @ A_hat, y[task_range[t]]).coef_
    elif link == 'logistic':
        for t in range(T):
            theta_hat[:, t] = LogisticRegression(fit_intercept = False).fit(x[task_range[t], :] @ A_hat, y[task_range[t]]).coef_
        
    # calculate the estimate of coef
    for t in range(T):
        beta_hat_step1[:, t] = A_hat @ theta_hat[:, t]
    
    beta_hat_step1 = torch.tensor(beta_hat_step1, requires_grad=False)
    
    if info:
        print("Step 1 is completed.\n", flush = True)
    
    ## Step 2: Biased regularization
    # torch initialization
    y = torch.tensor(y, requires_grad=False)
    x = torch.tensor(x, requires_grad=False)
    
    beta = torch.zeros((p, T), requires_grad=True, dtype=torch.float64)
    gamma = np.sqrt(p+np.log(T))*C2
    if link == 'linear':
        def ftotal2(beta):
           s = 0
           for t in range(T):
               s = s + 1/(2*n[t])*torch.dot(y[task_range[t]] - x[task_range[t], :] @ beta[:, t], y[task_range[t]] - x[task_range[t], :] @ beta[:, t]) + gamma/np.sqrt(n[t])*torch.norm(beta[:, t] - beta_hat_step1[:, t])
           return(s)
    elif link == 'logistic':
        def ftotal2(beta):
           s = 0
           for t in range(T):
               logits = torch.matmul(x[task_range[t], :], beta[:, t])
               s = s + 1/n[t]*torch.dot(1-y[task_range[t]], logits) + 1/n[t]*torch.sum(torch.log(1+torch.exp(-logits))) + gamma/np.sqrt(n[t])*torch.norm(beta[:, t] - beta_hat_step1[:, t])  
           return(s)
                         
    optimizer2 = optim.Adam([beta], lr=lr)
    loss_last = 1e8
    for i in range(max_iter):
        # Zero the gradients
        optimizer2.zero_grad()
        
        # Compute the loss (sum of the largest singular values)
        loss2 = ftotal2(beta)
        
        # Backward pass to compute gradients
        loss2.backward()
        
        # Update the matrices
        optimizer2.step()
        
        # Print the loss every 100 iterations
        if info:
            if (i + 1) % 100 == 0:
                print("Iteration {}/{}, Loss: {}".format(i+1, max_iter, loss2.item()), flush = True)
        if abs(loss_last-loss2.item())/loss2.item() <= tol:
            if info:
                print("Already converged. Stopped early.", flush = True)
            break
        loss_last = loss2.item()
        
    beta_hat_step1 = beta_hat_step1.numpy()
    beta_hat_step2 = beta.detach().numpy()
    if info:
        print("Step 2 is completed.\n", flush = True)
    if info:
        print("spectral stops running...", flush = True)
        
    return({"step1": beta_hat_step1, "step2": beta_hat_step2})


## Method-of-moments
def MoM(data, r):
    T = len(data)
    n = np.array([x.shape[0] for (x,y) in data])
    p = data[0][0].shape[1]
    n_total = np.sum(n)
    
    ## initialization
    x = np.zeros((n_total, p))
    y = np.zeros(n_total)
    
    # calculate sample indices for each task
    task_range = []
    start_index = 0
    for t in range(T):
        task_range.append(range(start_index, start_index+n[t]))
        start_index += n[t]
    
    # stack the x and y arrays
    for t in range(T):
        x[task_range[t], :] = data[t][0]
        y[task_range[t]] = data[t][1]
        
    M = (x.T @ np.diag(y**2) @ x)/n_total
    
    # SVD
    A_hat = np.linalg.svd(M)[0][:, 0:r]
    
    # calculate the theta estimate based on A_hat
    theta_hat = np.zeros((r, T))
    for t in range(T):
        theta_hat[:, t] = LinearRegression(fit_intercept = False).fit(x[task_range[t], :] @ A_hat, y[task_range[t]]).coef_
    
    # calculate the estimate of coef
    beta_hat = np.zeros((p, T))
    for t in range(T):
        beta_hat[:, t] = A_hat @ theta_hat[:, t]
        
    return(beta_hat)

