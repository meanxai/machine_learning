# [MXML-3-04] 5.boston(tls).py
# prediction of Boston house price by TLS
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/yDdbC9BhdwM
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
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Apply mean-centering to the training data
x_offset = x_train.mean(axis=0)
y_offset = y_train.mean()

xm_train = x_train - x_offset
ym_train = y_train - y_offset

# Apply Ridge regularization
REG_CONST = 0.01

# Cost function for OLS
def ols_loss(W):
    err = np.dot(W, xm_train.T) - ym_train
    mse = np.sqrt(np.mean(np.square(err)))
    loss = mse + REG_CONST * np.sum(np.square(W))
    return loss

# Cost function for TLS
def tls_loss(W):
    numerator = np.square(np.dot(W, xm_train.T) - ym_train)
    denominator = np.sum(np.square(W)) + 1
    d2 = numerator / denominator
    msd = np.sqrt(np.mean(d2))
    loss = msd + REG_CONST * np.sum(np.square(W))
    
    # save loss history
    trc_loss_train.append(loss)
    return loss

# Perform optimization process
trc_loss_train = []

# Perform OLS
W0 = np.array([1.0] * x_train.shape[1])  # W의 초깃값
result = optimize.minimize(ols_loss, W0)

# Perform TLS
# The optimal W found by OLS is used as the initial value of TLS.
W0 = result.x
result = optimize.minimize(tls_loss, W0)
print(result.success)    # check if success = True
print(result.message)

# Check the loss history
plt.figure(figsize=(6, 4))
plt.plot(trc_loss_train, label = 'loss_train')
plt.legend()
plt.xlabel('epochs')
plt.show()

# y_hat = coef * x + intercept
coef = result.x
intercept = y_offset - np.dot(x_offset, coef.T)

# Predict the y values of the test data
y_pred = np.dot(coef, x_test.T) + intercept

# Visually check the actual and predicted y values ​​of the test data.
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, s=20, c='r')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print('\n', df.head(10))

# Check the R2 score
print('\nTLS R2 score = {:.4f}'.format(r2_score(y_test, y_pred)))

# Check the R2 score from OLS
ols_coef = W0
ols_icept = y_offset - np.dot(x_offset, ols_coef.T)
y_ols_pred = np.dot(ols_coef, x_test.T) + ols_icept
print('OLS R2 score = {:.4f}'.format(r2_score(y_test, y_ols_pred)))

    