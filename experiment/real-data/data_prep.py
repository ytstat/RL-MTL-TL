#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation for HAR dataset
"""

import numpy as np
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(0)


# replace the dataset path of the folder with your own
dataset_path = '/Users/yetian/Library/CloudStorage/Dropbox/Columbia/Research/Project/MTL:TL-GMM/datasets/UCI HAR Dataset/'

x_train = pd.read_csv(dataset_path + 'train/X_train.txt', delimiter=r'\s+', header=None)
y_train = pd.read_csv(dataset_path + 'train/y_train.txt', delimiter=r'\s+', header=None)
subject_train = pd.read_csv(dataset_path + 'train/subject_train.txt', delimiter=r'\s+', header=None)

x_test = pd.read_csv(dataset_path + 'test/X_test.txt', delimiter=r'\s+', header=None)
y_test = pd.read_csv(dataset_path + 'test/y_test.txt', delimiter=r'\s+', header=None)
subject_test = pd.read_csv(dataset_path + 'test/subject_test.txt', delimiter=r'\s+', header=None)

x = np.array(pd.concat([x_train, x_test], ignore_index=True))
y = np.array(pd.concat([y_train, y_test], ignore_index=True)).squeeze()
subject = np.array(pd.concat([subject_train, subject_test], ignore_index=True)).squeeze()

# transform y to 2 classes
y[np.isin(y, [1,2,3,5])] = 0
y[np.isin(y, [4,6])] = 1

# --------------------
# standardize X in each task

x_transformed = np.zeros((x.shape[0], x.shape[1]))
scaler = StandardScaler()
for t in np.unique(subject):
    index = np.where(subject == t)[0]
    x_standardized = scaler.fit_transform(x[index, :])
    x_transformed[index, :] = x_standardized


# save the standardized data to a pickle file
with open('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/Code/datasets/har_standardized.pkl', 'wb') as f:
    pickle.dump((x_transformed, y, subject), f)
    

