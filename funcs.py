# -*- coding: utf-8 -*-
from numpy.linalg import norm
import numpy as np

def avg_distance(beta_hat, beta):
    s = 0
    T = beta.shape[1]
    for t in range(T):
        s = s + norm(beta_hat[:, t]-beta[:, t])/T
    return(s)

def max_distance(beta_hat, beta, S = None):
    if S is None:
        T = beta.shape[1]
        S = [i for i in range(0, T)]
    
    s = np.zeros(T)
    for t in S:
        s[t] = norm(beta_hat[:, t]-beta[:, t])
    return(max(s))

def all_distance(beta_hat, beta):
    T = beta.shape[1]
    s = np.zeros(T)
    for t in range(T):
        s[t] = norm(beta_hat[:, t]-beta[:, t])
    return(s)


def prediction(beta_hat, test_data):
    y_pred = []
    t = 0
    for (x, y) in test_data:
        y_pred.append((x @ beta_hat[:, t] > 0).astype(int))
        t += 1
    return(y_pred)

def all_classification_error(y_pred, test_data):
    error = np.zeros(len(test_data))
    for t in range(len(test_data)):
        error[t] = np.mean(y_pred[t] != test_data[t][1])
    return(error)