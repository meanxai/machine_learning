# [MXML-1-01] 1.KNN(distance).py
#
# This code was used in the machine learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/HmunSzYCKtg
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# create dataset
x, y = make_blobs(n_samples=300, n_features=2, 
                centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                cluster_std=0.15, center_box=(-1., 1.))

# Visualize the dataset and class by color
plt.figure(figsize=(5, 5))
for i, color in enumerate(['red', 'blue', 'green']):
    p = x[y==i]
    plt.scatter(p[:, 0], p[:, 1], s=50, c=color, 
                label='y=' + str(i), alpha=0.5)
plt.legend()    
plt.show()

# split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)
K = 10  # the number of nearest neighbors

# 1. Calculate the distance between test and train data.
d_train = x_train[np.newaxis, :, :]  # add a D0 axis
d_test = x_test[:, np.newaxis, :]    # add a D1 axis
distance = np.sqrt(np.sum((d_train - d_test) ** 2, axis=2))

# 2. Find K nearest neighbors
i_near = np.argsort(distance, axis=1)[:, :K]
y_near = y_train[i_near]

# 3. majority voting
y_pred = np.array([np.bincount(p).argmax() for p in y_near])

# Measure the accuracy for test data
print('Accuracy = {:.4f}'.format((y_pred == y_test).mean()))
