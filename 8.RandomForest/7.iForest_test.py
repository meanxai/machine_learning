# [MXML-8-07] 7.iForest_test.py
# Implementation of Isolation Forest using ExtraTreeRegressor
# sklearn's IsolationForest library makes it easy to implement 
# Isolation Forest, but I used ExtraTreeRegressor to better understand 
# how it works.
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/JpZJoOTjMWU
# 
from sklearn.tree import ExtraTreeRegressor
import numpy as np

# simple dataset
x = np.array([2, 2.5, 3.8, 4.1, 10.5, 15.4], dtype=np.float32).reshape(-1, 1)
n = x.shape[0]   # the number of data points
n_trees = 10     # the number of trees in Isolation Forest

# H(i) is the harmonic number and it can be estimated
# by ln(i) + 0.5772156649 (Euler’s constant).
def H(n):
    return np.log(n) + 0.5772156649

# average path length of unsuccessful search in BST
def C(n):
    return 2 * H(n-1) - (2 * (n-1) / n)

hx = np.zeros(n)
for t in range(n_trees):
    # Create a tree using random split points
    model = ExtraTreeRegressor(max_depth=3, max_features=1)
    
    # Fit the model to training data. 
    # Since it is unsupervised learning and there is no target value, 
    # a binary tree is created by randomly generating target values.
    model.fit(x, np.random.uniform(size=n))
    
    leaf_id = model.apply(x)  # indices of leaf nodes
    
    # depth of each node, internal and external nodes.
    node_depth = model.tree_.compute_node_depths()
    
    # h(x): accumulated path length of data points
    hx += node_depth[leaf_id] - 1.0
    
    print('Tree',t,':', (hx / (t+1)).round(1))

Ehx = hx / n_trees          # Average of h(x)
S = 2 ** (-(Ehx / C(n)))    # Anomaly scores for each data point
i_out = np.argsort(S)[-2:]  # Top 2 anomaly scores
outliers = x[i_out]         # outliers

print('\nAnomaly scores:')
print(S.round(3))
print('\nOutliers:')
print(outliers)

# import matplotlib.pyplot as plt
# from sklearn import tree

# plt.figure(figsize=(12, 8))
# tree.plot_tree(model)
# plt.show()
