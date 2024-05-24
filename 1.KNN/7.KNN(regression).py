# [MXML-1-06] 7.KNN(regression).py
#
# This code was used in the machine learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/XzTpLpqkepc
#
import numpy as np
import matplotlib.pyplot as plt

# Draw data (x) and estimation curves (y).
def plot_prediction(x, y, x_test, y_pred):
    plt.figure(figsize=(5,4))
    plt.scatter(x, y, c='blue', s=20, alpha=0.5, label='train data')
    plt.plot(x_test, y_pred, c='red', lw=2.0, label='prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 7)
    plt.legend()
    plt.show()

# Create nonlinear data for regression test
def noisy_sine_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.random()
       y = 2.0 * np.sin(2.0 * np.pi * x)+np.random.normal(0.0,s)+3.
       rtn_x.append(x)
       rtn_y.append(y)
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)

# Create training and test data
x, y = noisy_sine_data(n=1000, s=0.5)    
x = x.reshape(-1, 1)
x_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

# Normalize
x_mu = x.mean()
x_sd = x.std()
y_mu = y.mean()
y_sd = y.std()
zx_train = (x - x_mu) / x_sd
zx_test = (x_test - x_mu) / x_sd
zy_train = (y - y_mu) / y_sd

# Calculate distance between test and training data.
d_train = zx_train[np.newaxis, :, :]  # expand D0 axis
d_test = zx_test[:, np.newaxis, :]    # expand D1 axis
dist= np.sqrt(np.sum((d_train - d_test)**2, axis=2)) # (100, 1000)

# Find K nearest neighbors
K = 20
i_near = np.argsort(dist, axis=1)[:, :K]     # (100, 20)
y_near = zy_train[i_near]                    # (100, 20)

# Estimate the normalized y-values of the test data
# and restore them back to their original scale.
y_pred1 = y_near.mean(axis=1) * y_sd + y_mu  # (100,)

# Observe the results with the naked eye.
plot_prediction(x, y, x_test, y_pred1)

# Compare with the result of KNeighborsRegressor in sklearn.
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=K, weights='uniform')
knn.fit(zx_train, zy_train)                   # training
y_pred2 = knn.predict(zx_test) * y_sd + y_mu  # estimating
plot_prediction(x, y, x_test, y_pred2)

