# [MXML-1-07] 9.KNN(regression).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/_ZxTTvbZOtc
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Generate training and test data
n_train = 1000  # the number of training data points
n_test = 100    # the number of test data points
x_train = np.random.random(n_train).reshape(-1, 1)
y_train = 2.0 * np.sin(2.0 * np.pi * x_train)\
             + np.random.normal(0.0, 0.5, size=(n_train,1))+3.
y_train = y_train.reshape(-1)
x_test = np.linspace(x_train.min(), x_train.max(), n_test)\
             .reshape(-1, 1)

# Generate the distance matrix between x_test and x_train
d_train = x_train[np.newaxis, :, :]
d_test = x_test[:, np.newaxis, :]
dist= np.abs(d_train - d_test).reshape(n_test, n_train)

# Find K nearest neighbors
K = 20
i_near = np.argsort(dist, axis=1)[:, :K]    # (100, 20)
y_near = y_train[i_near]                    # (100, 20)

# Predict the y values ​​of the test data by simple average method
y_pred1 = y_near.mean(axis=1)

# Plot the training and test data points with their predicted 
# y values ​​(y_pred1)
def plot_prediction(y_pred):
    plt.figure(figsize=(6,4))
    plt.scatter(x_train, y_train, c='blue', s=20, alpha=0.5, label='train data')
    plt.plot(x_test, y_pred, c='red', lw=3.0, label='prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 7)
    plt.legend()
    plt.show()
    
# Predict the y-values ​​of the test data using the simple 
# average method.
plot_prediction(y_pred1)

# Predict the y values ​​of the test data using scikit-learn's KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=K)
knn.fit(x_train, y_train)
y_pred2 = knn.predict(x_test)
plot_prediction(y_pred2)

