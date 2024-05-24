# [MXML-1-02] 2.KNN(optimal_k).py
#
# This code was used in the machine learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/GQe5W0xmm5s
#
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# create dataset
x, y = make_blobs(n_samples=600, n_features=2, 
                centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                cluster_std=0.15, center_box=(-1., 1.))

# Visualize the dataset and class by color
plt.figure(figsize=(5, 5))
for i, color in enumerate(['red', 'blue', 'green']):
    p = x[y==i]
    plt.scatter(p[:, 0], p[:, 1], s=20, c=color, 
                label='y=' + str(i), alpha=0.5)
plt.legend()    
plt.show()

# split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)
N = x_train.shape[0]
   
# Z-score Normalization.
# We don't need to normalize this data because the features
# scale is similar, but I did it for practicing.

# Calculate the mean and standard deviation from the training data 
# and apply them to the test data.
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
z_train = (x_train - mean) / std
z_test = (x_test - mean) / std

# The function for KNN algorithm
def knn_predict(test, k):
    # 1. Calculate the distance between test and training data.
    d_train = z_train[np.newaxis, :, :]  # expand D0 axis
    d_test = test[:, np.newaxis, :]      # expand D1 axis
    distance = np.sqrt(np.sum((d_train - d_test) ** 2, axis=2))
    
    # 2. Find K nearest neighbors
    i_nearest = np.argsort(distance, axis=1)[:, :k]  # index
    y_nearest = y_train[i_nearest]
    
    # 3. majority voting
    return np.array([np.bincount(p).argmax() for p in y_nearest])

# Measure the accuracy of the test data while changing k.
accuracy = []
k_vals = np.arange(1, 500, 3)
for k in k_vals:
    # Estimate class of test data and measure the accuracy
    y_pred = knn_predict(z_test, k)
    accuracy.append((y_pred == y_test).mean())

# Observe the change in accuracy according to the change in K.
plt.figure(figsize=(5, 3))
plt.plot(k_vals, accuracy, 'o-')
plt.axvline(x=np.sqrt(N), c='r', ls='--')
plt.ylim(0.8, 1)
plt.show()

# Create a large number of test data and check the rough decision
# boundary.
x_many = np.random.uniform(-0.5, 1.5, (1000, 2))
z_many = (x_many - mean) / std
y_many = knn_predict(z_many, k=int(np.sqrt(N)))

# Check the decision boundary
plt.figure(figsize=(5,5))
color = [['red', 'blue', 'green'][a] for a in y_many]
plt.scatter(x_many[:, 0], x_many[:, 1], s=100, c=color, alpha=0.3)
plt.scatter(x_train[:, 0], x_train[:, 1], s=80, c='black')
plt.scatter(x_train[:, 0], x_train[:, 1], s=10, c='yellow')
plt.xlim(-0.5, 1.0)
plt.ylim(-0.5, 1.0)
plt.show()

