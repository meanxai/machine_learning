# [MXML-1-04] 6.WKNN(iris).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/Lu6GAc4FYz8
#
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
x, y = load_iris(return_X_y=True)

# Split the dataset to training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)
N = x_train.shape[0]  # the number of training data points
T = x_test.shape[0]   # the number of test data points
C = np.unique(y)      # categories of y: [0, 1, 2]
K = int(np.sqrt(N))   # appropriate K value

# Z-score Normalization.
mean = x_train.mean(axis=0); std = x_train.std(axis=0)
z_train = (x_train - mean) / std
z_test = (x_test - mean) / std

# Predict the class of test data.
# 1. Compute the distance matrix between test and train data.
d_train = z_train[np.newaxis, :, :]
d_test = z_test[:, np.newaxis, :]
dist = np.sqrt(np.sum((d_train - d_test) ** 2, axis=2))
dist += 1e-8  # To prevent the distance from becoming 0

# 2. Find K nearest neighbors.
i_near = np.argsort(dist, axis=1)[:, :K]
y_near = y_train[i_near]

# 3. Compute the inverse distance
w_inv = 1. / np.array([dist[i, :][i_near[i, :]] for i in range(T)])

# 4. Predict the class of the test data using the weights of the 
#    inverse distance
y_pred1 = []
for i in range(T):
    iw_dist = [w_inv[i][y_near[i] == j].sum() for j in C]
    y_pred1.append(np.argmax(iw_dist / w_inv[i].sum()))
y_pred1 = np.array(y_pred1)
    
# Measure the accuracy on the test data.
accuracy = (y_test == y_pred1).mean()
print('\nAccuracy on test data = {:.3f}'.format(accuracy))

# Compare with the results of sklearn's KNeighborsClassifier.
from sklearn.neighbors import KNeighborsClassifier

# 'distance': weight points by the inverse of their distance. 
# in this case, closer neighbors of a query point will have 
# a greater influence than neighbors which are further away.
knn = KNeighborsClassifier(n_neighbors=K, weights='distance')
knn.fit(z_train, y_train)
y_pred2 = knn.predict(z_test)
accuracy = (y_test == y_pred2).mean()
print('Accuracy on test data (sklearn) = {:.3f}'.format(accuracy))

print('from scratch: y_pred1\n', y_pred1)
print('from sklearn: y_pred2\n', y_pred2)

(y_pred1 != y_pred2).sum()
