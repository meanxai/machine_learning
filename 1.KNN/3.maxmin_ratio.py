# [MXML-01-03] 3.maxmin_ratio.py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/qZ_6UAVnNMw
#
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA 

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', parser='auto')
x = np.array(mnist['data']) / 255

# Compute the distances between a single data point and all other 
# data points in a given data set.
def distance(data):
    # Randomly choose a single data point from the dataset.
    i = np.random.randint(0, data.shape[0])
    tp = data[i]

    # Remove the chosen data point from the dataset.
    xp = np.delete(data, i, axis=0)

    # Compute the distances between tp and xp
    d = np.sqrt(np.sum((xp - tp) ** 2, axis=-1))

    # Return the minimum distance and maximum distance
    return d.min(), d.max()

# Compute the average ratio of minimum to maximum distances 
# in a 784-dimensional feature space
r_maxmin = []
for i in range(10):
    dmin, dmax = distance(x)
    r_maxmin.append(dmax / dmin)
print("max-min ratio (p=784): {0:.2f}".format(np.mean(r_maxmin)))

# Compute the average ratio of minimum to maximum distances 
# in a 5-dimensional feature space
pca = PCA(n_components=5)
pca.fit(x) 
x_pca = pca.transform(x)

r_maxmin = []
for i in range(10):
    dmin, dmax = distance(x_pca)
    r_maxmin.append(dmax / dmin)
print("max-min ratio (p=5)  : {0:.2f}".format(np.mean(r_maxmin)))
