# [MXML-7-03] 3.kmeans(plus).py
import numpy as np
import random as rd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate training data points.
x, y = make_blobs(n_samples=300, n_features=2, 
                  centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                  cluster_std=0.1, center_box=(-1., 1.))

N = x.shape[0]  # the number of data points
K = 3           # the number of data clusters
M = 10          # the number of data iterations

# Visualize the data points, x.
def plot_data(x):
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=30, c='black', alpha=0.5)
    plt.show()

# Visualize training data points and clusters color-coded.
def plot_cluster(x, cluster, centroid):
    plt.figure(figsize=(5, 5))
    color = [['red', 'blue', 'green'][a] for a in cluster]
    plt.scatter(x[:, 0], x[:, 1], s=30, c=color, alpha=0.5)
    plt.scatter(centroid[:, 0], centroid[:, 1], s=500, c='white')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=250, c='black')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=80, c='yellow')
    plt.show()

plot_data(x)

# Generate initial centroids using the K-Means++ algorithm.
xp = x.copy()
centroids = []
density = np.ones(xp.shape[0]) / N
for c in range(K):
    # (1) Choose an initial centroid c(1) uniformly at random from X 
    # (2) Choose the next centroid c(i), selecting c (i) = x' ∈ X with probability 
    idx = rd.choices(np.arange(xp.shape[0]), weights=density, k=1)[0]
    centroids.append(xp[idx])
    xp = np.delete(xp, idx, axis=0)

    # Create a distance matrix between data points xp and the centroids.
    # Please refer to the video [MXML-7-02] for how to create a distance matrix.
    x_exp = xp[np.newaxis, :, :]
    c_exp = np.array(centroids)[:, np.newaxis, :]
    dist = np.sqrt(np.sum(np.square(x_exp - c_exp), axis=2))
    
    # Find the centroid closest to each data point.
    assign = np.argmin(dist, axis=0)
    
    # Calculate D(x)
    # let D(x) denote the shortest distance from a data point x to 
    # the closest centroid we have already chosen
    Dx = np.sum(np.square(xp - np.array(centroids)[assign]), axis=1)
    
    # Create a probability density function to select the next centroid.
    density = Dx / np.sum(Dx)
    
centroids = np.array(centroids)

# Perform the K-Means algorithm using the centroids generated by K-Means++.
error = []
for m in range(M):
    # Calculate the distances between the training data points and the centroids. 
    x_exp = x[np.newaxis, :, :]
    c_exp = centroids[:, np.newaxis, :]
    dist = np.sqrt(np.sum(np.square(x_exp - c_exp), axis=2))
    
    # Assign each data point to the nearest centroid.
    assign = np.argmin(dist, axis=0)  # shape = (N,)
    
    # update centroids
    new_cent = []
    err = 0
    for c in range(K):
        # Find the data points assigned to centroid c.
        idx = np.where(assign == c)
        x_idx = x[idx]
        
        # To measure clustering performance, calculate the error.
        err += np.sum(np.sum(np.square(x_idx - centroids[c]), axis=1))
        
        # Compute the average coordinates of the data points
        # assigned to this centroid. And use that as new centroid.
        new_cent.append(np.mean(x_idx, axis=0))
    
    error.append(err)
    
    # Remove the if statement to see the centroids moving.
    if m == 0:
        plot_cluster(x, assign, centroids)
    
    # Update centroids
    centroids = np.array(new_cent)
            
# Visualize the training result.
plot_cluster(x, assign, centroids)

# Visualize error history
plt.plot(error, 'o-')
plt.title('final error =' + str(np.round(error[-1], 2)))
plt.show()

# Check the cluster number for each data point.
import pandas as pd
df = pd.DataFrame({'x1': x[:,0], 'x2': x[:,1], 'cluster': assign})
print(df.head(10))
