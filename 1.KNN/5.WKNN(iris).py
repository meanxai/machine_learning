# [MXML-1-04] 5.WKNN(iris).py
#
# This code was used in the machine learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/YtLOE_ggk3s
#
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y=True)  # Load iris dataset

# Creat training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)
N = x_train.shape[0]  # the number of training data
T = x_test.shape[0]   # the number of test data
C = np.unique(y)      # the number of class
K = int(np.sqrt(N))   # appropriate K

# Z-score Normalization.
mean = x_train.mean(axis=0); std = x_train.std(axis=0)
z_train = (x_train - mean) / std
z_test = (x_test - mean) / std

# Estimate the class of test data.
# 1. Calculate the distance between test and train data.
d_train = z_train[np.newaxis, :, :]  # Expand D0 axis
d_test = z_test[:, np.newaxis, :]    # Expand D1 axis
dist = np.sqrt(np.sum((d_train - d_test) ** 2, axis=2))
dist += 1e-8  # Prevents distance from becoming zero.

# 2. Find K nearest neighbors.
i_near = np.argsort(dist, axis=1)[:, :K]  # index
y_near = y_train[i_near]

# 3. Calculate inverse weighted distance
w_inv = 1. / np.array([dist[i, :][i_near[i, :]] for i in range(T)])

# 4. Estimate the class of test data by applying inverse weight
y_pred1 = []
for i in range(T):
    iw_dist = [w_inv[i][y_near[i] == j].sum() for j in C]
    y_pred1.append(np.argmax(iw_dist / w_inv[i].sum()))
y_pred1 = np.array(y_pred1)
   
# Measure the accuracy of test data.
accuracy = (y_test == y_pred1).mean()
print('\nAccuracy of test data = {:.3f}'.format(accuracy))

# Compare with the result of KNeighborsClassifier in sklearn.
from sklearn.neighbors import KNeighborsClassifier

# 'distance' : weight points by the inverse of their distance. 
# in this case, closer neighbors of a query point will have 
# a greater influence than neighbors which are further away.
knn = KNeighborsClassifier(n_neighbors=K, weights='distance')
knn.fit(z_train, y_train)
y_pred2 = knn.predict(z_test)
accuracy = (y_test == y_pred2).mean()
print('Accuracy of test data (sklearn) = {:.3f}\n'.format(accuracy))

print('from scratch: y_pred1\n', y_pred1)
print('from sklearn: y_pred2\n', y_pred2)


