# [MXML-12-01] 1.histogram-based.py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/N39NE4Nj6vc
# 
import numpy as np
from sklearn.datasets import make_blobs
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt

# Create a training data set.
x, y = make_blobs(n_samples=300, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.3]], 
                  cluster_std=0.15, center_box=(-1., 1.))

plt.figure(figsize=(4,4))
color = [['red', 'blue'][a] for a in y]
plt.scatter(x[:,0], x[:,1], c=color, alpha=0.3)
plt.show()

def find_local_split_point(f, s_point):
    GL = HL = 0.0
    l_bound = -np.inf           # lower left bound
    max_gain = -np.inf
    
    for j in s_point:
        # split the parent node into the left and right nodes.
        left = np.where(np.logical_and(f > l_bound, f <= j))[0]
        right = np.where(f > j)[0]
        
        # After splitting the parent node, calculate the scores of its children.
        GL += g[left].sum()
        HL += h[left].sum()
        GR = G - GL
        HR = H - HL
        
        # Calculate the gain for this split
        gain = (GL ** 2)/(HL + r) + (GR ** 2)/(HR + r) - p_score
            
        # Find the maximum gain.
        if gain > max_gain:
            max_gain = gain
            b_point = j      # best split point
        l_bound = j
    
    return b_point, max_gain

y0 = np.ones(shape=y.shape) * 0.5  # initial prediction
g = -(y - y0)            # negative residual.
h = y0 * (1. - y0)       # Hessian.

# Create a histogram of the parent node for each feature
n_bin = 30  # the number of bins
g0_parent, f0_bin = np.histogram(x[:, 0], n_bin, weights=g)  # feature 0
g1_parent, f1_bin = np.histogram(x[:, 1], n_bin, weights=g)  # feature 1

# Find the best split point of each feature
G = g.sum()
H = h.sum()
r = 0.01
p_score = (G ** 2) / (H + r)    # parent's score before splitting the node

# Find global best split point through parallel processing
# vertical partitioning method is used.
mp = Pool(2)
args = [[x[:, 0], f0_bin], [x[:, 1], f1_bin]]
ret = mp.starmap_async(find_local_split_point, args)
mp.close()
mp.join()

results = ret.get()
p1 = results[0][0];    p2 = results[1][0]
gain1 = results[0][1]; gain2 = results[1][1]

if gain1 > gain2:
    b_fid = 0
    b_point = p1
else:    
    b_fid = 1
    b_point = p2
    
print('\nbest feature id =', b_fid)
print('best split point =', b_point.round(3))

