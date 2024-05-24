# [MXML-01-03] 3.sklearnKNN(iris).py
#
# This code was used in the machine learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/yixEw8sL8x0
#
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load iris dataset
# x: data, the number of samples=150, the number of features=4
# y: target data with class (0,1,2)
x, y = load_iris(return_X_y=True)

# Create train, validation, test data
x_train, x_test, y_train, y_test=train_test_split(x, y, \
                                         test_size = 0.4)
x_val, x_test, y_val, y_test=train_test_split(x_test, y_test,\
                                         test_size = 0.5)
# Normalize data
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std # Z-score normalization
x_val = (x_val - mean) / std     # use mean and std from x_train
x_test = (x_test - mean) / std   # use mean and std from x_train

# Set K to sqrt(N)
sqr_k = int(np.sqrt(x_train.shape[0]))

# Build a KNN model
knn = KNeighborsClassifier(n_neighbors=sqr_k, metric='minkowski', p=2)

# Model fitting. Since KNN is a lazy learner, actual learning is not
# performed in this process. In the predict() process below, actual 
# learning is performed when test or validation data are given.
knn.fit(x_train, y_train)

# Estimate the class of validation date.
y_pred = knn.predict(x_val)

# Measure the accuracy for validation data
accuracy = (y_val == y_pred).mean()
print('\nK: sqr_K = {}, Accuracy for validation data = {:.3f}'\
      .format(sqr_k, accuracy))

# Determine the optimal K.
# Measure the accuracy of the validation data while changing K.
accuracy = []
for k in range(2, 20):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    accuracy.append((y_val == y_pred).mean())
    
# Find the K value when the accuracy is highest.
opt_k = np.array(accuracy).argmax() + 2

# Observe the change in accuracy for the change in K with the 
# naked eye.
plt.plot(np.arange(2, 20), accuracy, marker='o')
plt.xticks(np.arange(2, 20))
plt.axvline(x = opt_k, c='blue', ls = '--')
plt.axvline(x = sqr_k, c='red', ls = '--')
plt.ylim(0.7, 1.1)
plt.title('optimal K = ' + str(opt_k))
plt.show()

# Measure final performance with test data.
knn = KNeighborsClassifier(n_neighbors = opt_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = (y_test == y_pred).mean()
print('K: opt_k = {}, Accuracy for test data = {:.3f}'
      .format(opt_k, accuracy))

