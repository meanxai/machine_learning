# [MXML-1-02] 2.KNN(optimal_k).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/tIKsjeyaVnc
#
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# create dataset
x, y = make_blobs(n_samples=900, n_features=2, 
                centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                cluster_std=0.2, center_box=(-1., 1.))

# Visualize the dataset and classes by color
plt.figure(figsize=(5, 5))
for i, color in enumerate(['red', 'blue', 'green']):
    p = x[y==i]
    plt.scatter(p[:, 0], p[:, 1], s=20, c=color, 
                label='y=' + str(i), alpha=0.5)
plt.legend()    
plt.show()

# Split the dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
N = x_train.shape[0]
   
# Z-score Normalization.
# The values ​​in this data set have similar scales, 
# so there is no need to normalize them. But let's try this 
# just for practice.

# Calculate the mean and standard deviation from the training data 
# and apply them to the test data.
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
z_train = (x_train - mean) / std
z_test = (x_test - mean) / std

# A function for performing the KNN classification algorithm.
def knn_predict(train, test, k):
    # 1. Create a distance matrix.
    d_train = train[np.newaxis, :, :]   # Add a new axis at D0
    d_test = test[:, np.newaxis, :]     # Add a new axis at D1
    
    p = 2  # Euclidean distance
    d = np.sum(np.abs(d_train - d_test) ** p, axis=-1) ** (1/p)
    
    # 2. Find K nearest neighbors
    i_nearest = np.argsort(d, axis=1)[:, :k]  # index
    y_nearest = y_train[i_nearest]
    
    # 3. majority voting
    return np.array([np.bincount(i).argmax() for i in y_nearest])

# Measure the accuracy of the test data while changing K value.
accuracy = []
k_vals = np.arange(1, 700, 10)
for k in k_vals:
    # Estimate the classes of all test data points and measure the accuracy.
    y_pred = knn_predict(z_train, z_test, k)
    accuracy.append((y_pred == y_test).mean())
    
# Observe how the accuracy changes as K changes.
plt.figure(figsize=(5, 3))
plt.plot(k_vals, accuracy, '-')
plt.axvline(x=np.sqrt(N), c='r', ls='--')
plt.ylim(0.5, 1)
plt.show()

# Generate a large number of test data points and roughly determine 
# the decision boundary.
# x_many = np.random.uniform(-0.5, 1.5, (1000, 2))
x_many = np.random.uniform(-0.5, 1.5, (1000, 2))
z_many = (x_many - mean) / std
y_many = knn_predict(z_train, z_many, k=int(np.sqrt(N)))

# Check the decision boundary
plt.figure(figsize=(5,5))
color = [['red', 'blue', 'green'][a] for a in y_many]
plt.scatter(x_many[:, 0], x_many[:, 1], s=100, c=color, alpha=0.3)
plt.scatter(x_train[:, 0], x_train[:, 1], s=80, c='black')
plt.scatter(x_train[:, 0], x_train[:, 1], s=10, c='yellow')
plt.xlim(-0.5, 1.0)
plt.ylim(-0.5, 1.0)
plt.show()

