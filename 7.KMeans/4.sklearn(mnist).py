# [MXML-7-03] 4.sklearn(mnist).py
# MNIST clustering
# This code can be found at github.com/meanxai/machine_learning.
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784')
# mnist.pkl is the saved mnist.
with open('data/mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)

# Use only 10,000 data points and normalize them between 0 and 1
x = np.array(mnist['data'][:10000]) / 255.

# Cluster the data points into 10 groups using K-Means++.
model = KMeans(n_clusters=10, 
               init='k-means++',   # default
               max_iter = 50, 
               n_init = 5)

model.fit(x)
clust = model.predict(x)
centroids = model.cluster_centers_
   
# Check out the images for each cluster.
for k in np.unique(clust):
    # Find 10 images belonging to cluster k, and centroid image.
    idx = np.where(clust == k)[0]
    images = x[idx[:10]]
    centroid = centroids[k, :]
    
    # Find 10 images closest to each centroid image.
    # d = np.sqrt(np.sum((x[idx] - centroid)**2, axis=1))
    # nearest = np.argsort(d)[:10]
    # images = x[idx[nearest]]
    
    
    # display the central image
    f = plt.figure(figsize=(8, 2))
    image = centroid.reshape(28, 28)
    ax = f.add_subplot(1, 11, 1)
    ax.imshow(image, cmap=plt.cm.bone)
    ax.grid(False)
    ax.set_title("C")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.tight_layout()
    
    # display 10 images belonging to the centroid
    for i in range(10):
        image = images[i].reshape(28,28)
        ax = f.add_subplot(1, 11, i + 2)
        ax.imshow(image, cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(k)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
