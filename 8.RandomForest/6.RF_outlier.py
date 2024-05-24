# [MXML-8-06] 6.RF_outlier.py
# Outlier detection using Random Forestâ€™s proximity matrix
# Reference [2]:
# https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#outliers
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/ps2QXPnPHVM
# 
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate training data
x, y = make_blobs(n_samples=600, n_features=2, 
                centers=[[0., 0.], [0.5, 0.5]], 
                cluster_std=0.2, center_box=(-1., 1.))

# Create Proximity matrix
# normalize = 0: pm / n_tree
# normalize â‰  0: Normalize columns to sum to 1
def proximity_matrix(model, x, normalize=0):
    n_tree = len(model.estimators_)
    
    # Apply trees in the forest to X, return leaf indices.
    leaf = model.apply(x)  # shape = (x.shape[0], n_tree)
    
    pm = np.zeros(shape=(x.shape[0], x.shape[0]))
    for i in range(n_tree):
        t = leaf[:, i]
        p = np.equal.outer(t, t) * 1.
        pm += p

    np.fill_diagonal(pm, 0)    
    if normalize == 0:
        return pm / n_tree
    else:
        return pm / pm.sum(axis=0, keepdims=True)

n_estimators = 50
n_depth = 5

# Detect outliers using a proximity matrix
model = RandomForestClassifier(n_estimators=n_estimators,
                               max_depth=n_depth,
                               max_features="sqrt",  # default
                               bootstrap=True,       # default
                               oob_score=True)
model.fit(x, y)

# Create a proximity matrix
pm = proximity_matrix(model, x, normalize=0)

i_y0 = np.where(y == 0)[0]
i_y1 = np.where(y == 1)[0]
i_y = [i_y0, i_y1]

# 1) average proximity
pi_bar = []
for i in range(pm.shape[0]):
    j_class = y[i]        # the class of data instance i
    j_same = i_y[j_class] # Data point IDs with the same class as data point i
    pi_bar.append(np.sum(pm[i, j_same] ** 2))

# 2) raw outlier measure
o_raw = x.shape[0] / np.array(pi_bar)

# 3) final outlier measure
# For convenience of coding, the mean value was used instead of the 
# median, and the standard deviation was used instead of the absolute 
# deviation.
f_measure = []
for i in range(o_raw.shape[0]):
    j_class = y[i]         # the class of the data instance i
    j_same = i_y[j_class]  # Data point IDs with the same class as data point i
    f_measure.append((o_raw[i] - o_raw[j_same].mean()) / o_raw[j_same].std())

# Data in the upper top_rate percentage of f_measure are considered outliers.
top_rate = 0.07  # top 5%
top_idx = np.argsort(f_measure)[::-1][:int(top_rate * x.shape[0])]

# Visualize normal data and outliers by color.
plt.figure(figsize=(7, 7))
color = [['blue', 'red'][i] for i in y]
color_out = [['blue', 'red'][i] for i in y[top_idx]]
plt.scatter(x[:, 0], x[:, 1], s=30, c=color, alpha=0.5)
plt.scatter(x[top_idx, 0], x[top_idx, 1], s=400, c='black', alpha=0.5)  # outlier scatter
plt.scatter(x[top_idx, 0], x[top_idx, 1], s=200, c='white')
plt.scatter(x[top_idx, 0], x[top_idx, 1], s=30, c=color_out)
plt.show()
