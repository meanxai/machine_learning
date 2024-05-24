# [MXML-7-02] 1.kmeans(basic).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/hToqjr5Kx4Q
# 
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time

# Generate training data
x, y = make_blobs(n_samples=300, n_features=2, 
                  centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                  cluster_std=0.1, center_box=(-1., 1.))
                  
N = x.shape[0]  # the number of data points
K = 3           # the number of clusters
M = 10          # the number of iterations
L = 10          # the number of attempts to prevent local minimum problem.

# Visualize training data
def plot_data(x):
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=30, c='black', alpha=0.5)
    plt.show()

# Visualize training data and clusters color-coded.
def plot_cluster(x, cluster, centroid):
    plt.figure(figsize=(5, 5))
    color = [['red', 'blue', 'green'][a] for a in cluster]
    plt.scatter(x[:, 0], x[:, 1], s=30, c=color, alpha=0.5)
    plt.scatter(centroid[:, 0], centroid[:, 1], s=500, c='white')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=250, c='black')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=80, c='yellow')
    plt.show()

plot_data(x)

f_error = [9999]      # final error history
f_centroids = None    # final centroids
f_assign = None       # final assignment
for l in range(L):    # Repeat L times, changing the position of the centroids.
    # Initialize the centroids
    # Randomly select K data points and use them as the K initial centroids.
    idx = np.random.choice(np.arange(N), K)
    centroids = x[idx]
    error = []
    for m in range(M):
        # Calculate the distances between the training data points and the centroids.
        x_exp = x[np.newaxis, :, :]            # add a D0 axis. (1, N, 2)
        c_exp = centroids[:, np.newaxis, :]    # add a D1 axis. (K, 1, 2)
        
        # create the distance matrix using matrix broadcasting.
        # The shape of dist = (K, N)
        dist = np.sqrt(np.sum(np.square(x_exp - c_exp), axis=2))
    
        # Assign each data point to the nearest centroid.
        # if assign = [0 1 2 1 0 2 0 1 ...]
        # The first data point is assigned to cluster 0, 
        # and the second data point is assigned to cluster 1.
        assign = np.argmin(dist, axis=0)  # shape = (N,)
        
        # update centroids
        new_cent = []
        err = 0
        for c in range(K):
            # Find the data points assigned to centroid c.
            idx = np.where(assign == c)
            x_idx = x[idx]
            
            # The error is measured as the sum of the squares of 
            # the distances between data points and their centroid.
            err += np.sum(np.sum(np.square(x_idx - centroids[c]), axis=1))
            
            # Compute the average coordinates of the data points
            # assigned to this centroid. And use that as new centroid.
            new_cent.append(np.mean(x_idx, axis=0))
        
        error.append(err)
        
        # To observe the centroid moving, set L=1 and run the code below.
        # plot_cluster(x, assign, centroids)
        # print("iteration:", m)
        # time.sleep(1)
        
        # Update centroids
        centroids = np.array(new_cent)
    
    # Among the L number of iterations, the one with the smallest error 
    # is selected as the final result.
    if error[-1] < f_error[-1]:
        f_error = np.copy(error)
        f_centroids = np.copy(centroids)
        f_assign = np.copy(assign)
        
# Visualize the training result.
plot_cluster(x, f_assign, f_centroids)

# Visualize error history
plt.plot(f_error, 'o-')
plt.title('final error =' + str(np.round(error[-1], 2)))
plt.show()

# Check the cluster number for each data point.
import pandas as pd
df = pd.DataFrame({'x1': x[:,0], 'x2': x[:,1], 'cluster': f_assign})
print(df.head(10))
