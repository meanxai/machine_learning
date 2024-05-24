# [MXML-3-5] 7.lwr(sklearn).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/d1-QS4uTgj8
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Generate sinusoidal data with Gaussian noise added.
def noisy_sine_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x= np.random.random()
       y= 2.0*np.sin(2.0*np.pi*x)+np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)

# Create 1,000 data points for LWR testing.
x, y = noisy_sine_data(n=1000, s=0.7)
x_train, x_test, y_train, y_test = train_test_split(x, y)
x1_train = np.hstack([np.ones([x_train.shape[0], 1]), x_train])

# Visualize the training and test data
plt.figure(figsize=(6, 5))
plt.scatter(x_train, y_train, s=5, c='orange', label='train')
plt.scatter(x_test, y_test, marker='+', s=30, c='blue', 
            label='test')
plt.legend()
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axhline(y=0, ls='--', lw=0.5, c='black')
plt.show()

# Find the weight for each data point.
# train: training data, test: test data point to be predicted
def get_weight(train, test, tau):
    d2 = np.sum(np.square(train - test), axis=1)
    w = np.exp(-d2 / (2. * tau * tau))
    return w 

# predict the target value of the test data
y_pred = []
for tx in x_test:
    weight = get_weight(x_train, tx, 0.05)
    model = Ridge(alpha=0.01)
    model.fit(x_train, y_train, sample_weight = weight)
    y_pred.append(model.predict(tx.reshape(-1,1))[0])
y_pred = np.array(y_pred).reshape(-1,)

# Visualize the predicted results
plt.figure(figsize=(6, 5))
plt.scatter(x_train, y_train, s=5, c='orange', label='train')
plt.scatter(x_test, y_test, marker='+', s=30, c='blue',
            label='test')
plt.scatter(x_test, y_pred, s=5, c='red', label='prediction')
plt.legend()
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axhline(y=0, ls='--', lw=0.5, c='black')
plt.show()
