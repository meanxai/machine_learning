# [MXML-11-08] 4.approximation(2).py
# Tianqi Chen et, al., 2016, XGBoost: A Scalable Tree Boosting System
# 3. SPLIT FINDING ALGORITHMS
# 3.3 Weighted Quantile Sketch
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/ejUvX1L-yzE
# 
import numpy as np
from sklearn.datasets import make_blobs
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time

# Create a simple training dataset
x, y = make_blobs(n_samples=500000, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.5]], 
                  cluster_std=0.2, center_box=(-1., 1.))

x_train, x_test, y_train, y_test = train_test_split(x, y)

TREES = 200  # the number of trees
DEPTH = 5    # the depth of tree
ETA = 0.1    # learning rate, eta
LAMB = 1.0   # regularization constant
GAMMA = 0.1  # pruning constant
EPS = 0.03   # epsilon for approximate and weighted quantile sketch

# 1. Exact Greedy Algorithm (EGA)
# -------------------------------
start_time = time.time()
model = XGBClassifier(n_estimators = TREES,
                      max_depth = DEPTH,
                      learning_rate = ETA,    # η
                      gamma = GAMMA,          # γ for pruning
                      reg_lambda = LAMB,      # λ for regularization
                      base_score = 0.5,       # initial prediction value
                      tree_method = 'exact')  # exact greedy algorithm

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print('\nExact greedy algorithm:')
print('Accuracy =', np.round(acc, 3))
print('running time = {:.2f} seconds'.format(time.time() - start_time))

# 2.Approximate Algorithm (AA).
# -------------------------------
start_time = time.time()
model = XGBClassifier(n_estimators = TREES,
                      max_depth = DEPTH,
                      learning_rate = ETA,    # η
                      gamma = GAMMA,          # γ for pruning
                      reg_lambda = LAMB,      # λ for regularization
                      base_score = 0.5,       # initial prediction value
                      max_bin = int(1/EPS),   # sketch_eps is replaced by max_bin
                      tree_method = 'approx') # weighted quantile sketch

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print('\nWeighted Quantile Sketch:')
print('Accuracy =', np.round(acc, 3))
print('running time = {:.2f} seconds'.format(time.time() - start_time))

# tree_method:
#
# https://xgboost.readthedocs.io/en/stable/parameter.html
# auto: Same as the hist tree method.
# exact: Exact greedy algorithm. Enumerates all split candidates.
# approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
# hist: Faster histogram optimized approximate greedy algorithm.
#
# https://xgboost.readthedocs.io/en/latest/treemethod.html
# approx tree method: An approximation tree method described in 
# reference paper. It runs sketching before building each tree using 
# all the rows (rows belonging to the root). Hessian is used as weights 
# during sketch. The algorithm can be accessed by setting tree_method 
# to approx.

# max_bin:
#
# https://github.com/dmlc/xgboost/issues/8063
# Also, the parameter sketch_eps is replaced by max_bin for aligning 
# with hist, the old default for max_bin translated from sketch_eps 
# was around 63 while the rewritten one is 256, which means the new 
# implementation builds larger histogram.

# import matplotlib.pyplot as plt
# x, y = make_blobs(n_samples=10000, n_features=2, 
#                   centers=[[0., 0.], [0.5, 0.5]], 
#                   cluster_std=0.2, center_box=(-1., 1.))

# plt.figure(figsize=(5,5))
# color = ['red' if a == 1 else 'blue' for a in y]
# plt.scatter(x[:, 0], x[:, 1], s=1, alpha=0.8, c=color)
# # plt.xlim(-0.5, 1.0)    
# # plt.ylim(-0.5, 1.0)
# plt.show()