# [MXML-7-02] 2.sklearn(kmeans).py
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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate training data
x, y = make_blobs(n_samples=300, n_features=2, 
                  centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                  cluster_std=0.1, center_box=(-1., 1.))

K = 3           # the number of clusters
M = 10          # the number of iterations
L = 10          # the number of attempts to prevent local minimum problem.

model = KMeans(n_clusters = K,    # the number of clusters
               init='random',     # randomly initialize centroids
               max_iter=M,        # max iterations
               n_init = L)        # Number of times the k-means algorithm 
                                  # is run with different centroid seeds.

model.fit(x)

# Visualize training data and clusters color-coded.
def plot_cluster(x, cluster, centroid):
    plt.figure(figsize=(5, 5))
    color = [['red', 'blue', 'green'][a] for a in cluster]
    plt.scatter(x[:, 0], x[:, 1], s=30, c=color, alpha=0.5)
    plt.scatter(centroid[:, 0], centroid[:, 1], s=500, c='white')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=250, c='black')
    plt.scatter(centroid[:, 0], centroid[:, 1], s=80, c='yellow')
    plt.show()

# Visualize the training result.
plot_cluster(x, model.labels_, model.cluster_centers_)

# print the final error
# Sum of squared distances of samples to their closest cluster center
print('\nerror = {:.4f}'.format(model.inertia_))

