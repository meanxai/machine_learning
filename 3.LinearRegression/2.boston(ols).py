# [MXML-3-03] 2.boston(ols).py
# prediction of Boston house price
# Applying Mean centering, Normalization, Ridge Regularization
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/gLekbL_pI1A
#
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle

# Read Boston house price dataset
with open('data/boston_house.pkl', 'rb') as f:
    data = pickle.load(f)
    
x = data['data']      # shape = (506, 13)
y = data['target']    # shape = (506,)

# Split the dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)
REG_CONST = 0.01   # regularization constant

# Mean centering & Normalization are performed on training data.
x_offset = x_train.mean(axis=0)
x_scale = x_train.std(axis=0)
y_offset = y_train.mean()

xm_train = (x_train - x_offset) / x_scale
ym_train = y_train - y_offset

# Regularized mean squared error loss function
def ols_loss(W):
    # Calculating MSE using the training data
    d_train = np.dot(W, xm_train.T) - ym_train
    mse = np.mean(np.square(d_train))
    loss = mse + REG_CONST * np.sum(np.square(W))
    
    # Save the loss history.
    trc_loss.append(loss)
    return loss

# Perform optimization process
trc_loss = []
W0 = np.ones(xm_train.shape[1]) * 0.1  # W의 초깃값.
result = optimize.minimize(ols_loss, W0)

# Check the results
print(result.success)    # check if success = True
print(result.message)

# Visually check the regularized MSE of the training data.
plt.figure(figsize=(6, 4))
plt.plot(trc_loss, label = 'loss_train')
plt.legend()
plt.xlabel('epochs')
plt.show()

# Convert result.x to the coef and the intercept
# y_hat = coef * x + intercept
coef = result.x / x_scale
intercept = y_offset - np.dot(x_offset, coef.T)

# Predict y values of the test data.
y_pred = np.dot(coef, x_test.T) + intercept

# Visually check the predicted and actual y values ​​of the test data.
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, s=20, c='r')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print('\n', df.head(10))

# Check R2 score of the test data.
print('\nR2 score = {:.4f}'.format(r2_score(y_test, y_pred)))

