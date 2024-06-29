#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:02:49 2024

@author: yetian
"""

import numpy as np
import torch
from torch.optim import LBFGS, SGD
from tqdm.notebook import trange, tqdm


def generate_inputs(d, k, n, T, eps):
    top_block_samples = np.sqrt(eps) * torch.randn(T, n, d-k)
    bottom_block_samples = torch.randn(T, n, k)
    return torch.cat([top_block_samples, bottom_block_samples], dim=-1)

def generate_weights(d, k, T, eps):
    dominant, residual = torch.randn(T, k), torch.randn(T, k)
    dominant /= torch.norm(dominant, dim=-1, keepdim=True)
    residual /= torch.norm(residual, dim=-1, keepdim=True)
    return torch.cat([np.sqrt(1/(2*eps)) * dominant, torch.zeros((T, d-2*k)), residual], axis=-1)

def generate_labels(inputs, predictors, noise_stddev):
    means = torch.matmul(inputs, predictors[:, :, None])[:, :, 0]
    return means + noise_stddev * torch.randn(*means.shape)


def train_loop(opt_vars, optimizer, loss_fn, projection, datasets, train_params, plot_losses=False):
    """
    opt_vars: list of PyTorch variables
    optimizer: pytorch.optim
    loss_fn: takes (opt_vars, X, y)
    projection: takes opt_vars, applies changes to .data only
    datasets: list containing two tensors of shapes [(T, n, d), (T, n)]
              first tensor is inputs, second is labels, T is number of parallel datasets
    train_params: {batch_size, num_batches, train_desc}
    """
    dataset_size = datasets[0].shape[1]
    batch_size = train_params["batch_size"]
    num_batches_per_pass = int(np.ceil(dataset_size / batch_size))
    
    with torch.no_grad():
        projection(opt_vars)
        
    best_opt_vars, best_loss = [opt_var.data.clone() for opt_var in opt_vars], np.inf
    losses = []
    for num_batch in trange(train_params["num_batches"], leave=False, desc=train_params["train_desc"]):
        batch_idx = num_batch % num_batches_per_pass
        
        # Check if this is a new_pass - if so, shuffle the data
        if batch_idx == 0:   
            perm = torch.randperm(dataset_size)
            datasets = [datasets[0][:, perm], datasets[1][:, perm]]
            
        start_idx, end_idx = batch_idx * batch_size, (batch_idx + 1) * batch_size
        batch_X, batch_y = datasets[0][:, start_idx:end_idx], datasets[1][:, start_idx:end_idx]

        optimizer.zero_grad()
        loss_fn(opt_vars, batch_X, batch_y).backward()
        optimizer.step(lambda: loss_fn(opt_vars, batch_X, batch_y))

        with torch.no_grad():
            projection(opt_vars)
            cur_loss = loss_fn(opt_vars, datasets[0], datasets[1]).item()
            if cur_loss <= best_loss:
                best_opt_vars = [opt_var.data.clone() for opt_var in opt_vars]
                best_loss = cur_loss
            losses.append(cur_loss)

    return best_opt_vars



def prep_adaptrep_training(d, k, T, n, eps, noise_stddev):
    B_train = torch.randn(d, k, requires_grad=True) 
    W_train = torch.randn(k, T, requires_grad=True)
    Delta_train = torch.randn(d, T, requires_grad=True)
    opt_vars = [B_train, W_train, Delta_train]
    optimizer = LBFGS(opt_vars, lr=0.1, history_size=5)

    def loss_fn(opt_vars, inputs, labels):
        B_train, W_train, Delta_train = opt_vars
        weights_train = (torch.matmul(B_train, W_train) + Delta_train).T
        predictions = torch.matmul(inputs, weights_train[:, :, None])[:, :, 0]
        mse = torch.mean(torch.square(labels - predictions))
        regularization = 1/(4*T) * torch.sum(torch.square(torch.matmul(B_train.T, B_train) - torch.matmul(W_train, W_train.T)))
        delta_reg = noise_stddev * (np.sqrt(k * d / (n * T)) + np.sqrt(k / n)) * torch.mean(torch.norm(Delta_train, dim=0))
        return mse + regularization + delta_reg

    def create_projection(C0):
        def projection(opt_vars):
            B_train, W_train, Delta_train = opt_vars
            W_train.data /= torch.max(torch.norm(W_train, dim=0, keepdim=True) / np.sqrt(C0 * np.sqrt(k / T)), torch.tensor([1.]))
            v_randn = (1e-2)*torch.randn(k, T)
            W_train.data = torch.where(torch.isnan(W_train.data), v_randn, W_train.data)
            W_train.data = torch.where(torch.isinf(W_train.data), v_randn, W_train.data)
            W_train.data /= torch.max(torch.svd(W_train, compute_uv=False)[1][0] / np.sqrt(C0 * np.sqrt(T / k)), torch.tensor([1.]))
            v_randn_b = (1e-2)*torch.randn(d, k)
            B_train.data = torch.where(torch.isnan(B_train.data), v_randn_b, B_train.data)
            B_train.data = torch.where(torch.isinf(B_train.data), v_randn_b, B_train.data)
            B_train.data /= torch.max(torch.svd(B_train, compute_uv=False)[1][0] / np.sqrt(C0 * np.sqrt(T / k)), torch.tensor([1.]))
            Delta_train.data /= torch.max(torch.norm(Delta_train, dim=0, keepdim=True) / 2, torch.tensor([1]).float())
        return projection
    
    return opt_vars, optimizer, loss_fn, create_projection(3) 


def find_adversary(B_learned):
    d, k = B_learned.shape
    orth_opt = torch.cat([torch.eye(k), torch.zeros(d-k, k)], axis=0)
    delta_opt = torch.cat([torch.zeros(d-k, k), torch.eye(k)], axis=0)
    
    orth_learned = torch.svd(B_learned, some=True)[0]
    projector_learned = torch.matmul(orth_learned, orth_learned.T)
    proj_perp_learned = torch.eye(projector_learned.shape[0]) - projector_learned
    
    opt_res = torch.matmul(proj_perp_learned, orth_opt)
    delta_res = torch.matmul(proj_perp_learned, delta_opt)
    
    return torch.matmul(orth_opt, torch.svd(opt_res, some=True)[2][:, :1]), \
           torch.matmul(delta_opt, torch.svd(delta_res, some=True)[2][:, :1])

def prep_training_w_rep(B_learned, d, k, T, n, eps, noise_stddev):
    W_train = torch.randn(k, T, requires_grad=True)
    Delta_train = torch.randn(d, T, requires_grad=True)
    opt_vars = [W_train, Delta_train]
    optimizer = SGD(opt_vars, lr=0.01)

    def loss_fn(opt_vars, inputs, labels):
        W_train, Delta_train = opt_vars
        weights_train = (torch.matmul(B_learned, W_train) + Delta_train).T
        predictions = torch.matmul(inputs, weights_train[:, :, None])[:, :, 0]
        mse = torch.sum(torch.mean(torch.square(labels - predictions), dim=-1))
        regularization = (2 * k / np.sqrt(n)) * torch.sum(torch.norm(Delta_train, dim=0))
        return mse + regularization

    def projection(opt_vars):
        pass
    
    return opt_vars, optimizer, loss_fn, projection

def prep_baseline(d, k, T, n, eps, noise_stddev):
    Delta_train = torch.randn(d, T, requires_grad=True)
    opt_vars = [Delta_train]
    optimizer = SGD(opt_vars, lr=0.1)
    
    def loss_fn(opt_vars, inputs, labels):
        Delta_train = opt_vars[0]
        weights_train = Delta_train.T
        predictions = torch.matmul(inputs, weights_train[:, :, None])[:, :, 0]
        return torch.sum(torch.mean(torch.square(labels - predictions), dim=-1))
    
    def projection(opt_vars):
        pass
    
    return opt_vars, optimizer, loss_fn, projection


def AdaptRep(data, r, num_batches = 200, batch_size = 2):
    T = len(data)
    p = data[0][0].shape[1]
    n = data[0][0].shape[0]
    x = np.zeros((T, n, p))
    y = np.zeros((T, n))
    noise_stddev = 1
    eps = r / p
    for t in range(T):
        x[t, :, :] = data[t][0]
        y[t, :] = data[t][1]
    
    y0 = torch.tensor(y, requires_grad=False, dtype=torch.float32)
    x0 = torch.tensor(x, requires_grad=False, dtype=torch.float32)
    

    opt_vars, optimizer, loss_fn, projector = prep_adaptrep_training(p, r, T, n, eps, noise_stddev)
    opt_vars = train_loop(opt_vars, optimizer, loss_fn, projector, (x0, y0), {"num_batches": num_batches, "batch_size": batch_size, "train_desc": "Training by adaptation"})

    beta_hat_AdaptRep = (opt_vars[0] @ opt_vars[1] + opt_vars[2]).detach().numpy()
    return(beta_hat_AdaptRep)


# n = 100; p = 30; r = 5; T = 25; h = 0; epsilon = 0
# noise_stddev = 1
# eps = r / p
# y0 = torch.tensor(y, requires_grad=False, dtype=torch.float32)
# x0 = torch.tensor(x, requires_grad=False, dtype=torch.float32)

# list_att = list()
# for att in range(10):
#     while True:
#         try:
#             opt_vars, optimizer, loss_fn, projector = prep_adaptrep_training(p, r, T, n, eps, noise_stddev)
#             opt_vars = train_loop(
#                 opt_vars, optimizer, loss_fn, projector, (x0, y0),
#                 {"num_batches": 500, "batch_size": 64, "train_desc": "Training by adaptation"}
#             )
#             break
#         except RuntimeError:
#             pass
    
#     list_att.append({"representation": opt_vars[0].numpy(),
#                      "k": r,
#                      "T": T,
#                      "d": p,
#                      "n": n,
#                      "eps": eps,
#                      "noise_stddev": noise_stddev})


# adapt_data = list_att
# Bs_adapt = [torch.svd(torch.from_numpy(data["representation"]), some=True)[0] for data in adapt_data]

# # Compute worst-case predictor for AdaptRep
# adv_dirs = [find_adversary(B_adapt) for B_adapt in Bs_adapt]
# adapt_advs = [np.sqrt(1 / (2 * eps)) * opt_dir + delta_dir for (opt_dir, delta_dir) in adv_dirs]

# beta_att = list()
# for att in range(10):
#     opt_vars, optimizer, loss_fn, projection = prep_training_w_rep(Bs_adapt[att], p, r, T, n, eps, noise_stddev)
#     W_train, Delta_train = train_loop(opt_vars, optimizer, loss_fn, projection, (x0, y0),{"num_batches": 3000, "batch_size": 16, "train_desc": "Training using AdaptRep-derived representation"})
#     estimator_delta = ((torch.matmul(Bs_adapt[att], W_train) + Delta_train).detach() - adapt_advs[att]).numpy()
#     beta_att.append(estimator_delta)
    
# opt_vars, optimizer, loss_fn, projector = prep_adaptrep_training(p, r, T, n, eps, noise_stddev)
# opt_vars = train_loop(opt_vars, optimizer, loss_fn, projector, (x0, y0), {"num_batches": 500, "batch_size": 64, "train_desc": "Training by adaptation"})


# B_adapt = opt_vars[0]

# opt_vars, optimizer, loss_fn, projection = prep_training_w_rep(B_adapt, p, r, T, n, eps, noise_stddev)
# W_train, Delta_train = train_loop(opt_vars, optimizer, loss_fn, projection, (x0, y0),{"num_batches": 3000, "batch_size": 16, "train_desc": "Training using AdaptRep-derived representation"})

# estimator_delta = (torch.matmul(B_adapt, W_train) + Delta_train).detach().numpy()

# for beta_t in beta_att:
#     print("AdaptRep: {}".format(max(all_distance(beta_t, beta)[S])))


