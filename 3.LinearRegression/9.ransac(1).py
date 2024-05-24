# [MXML-3-07] 9.ransac(1).py
# Implementing RANSAC from scratch
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/A2QnStjnlVE
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate n data samples with outliers.
def reg_data_outlier(a, b, n, s, outlier_rate=0.1):
    n1 = int(n * outlier_rate) # the number of outliers
    n2 = n - n1                # the number of inliers
    
    # Generate normal data points (inliers)
    x2 = np.random.normal(0.0, 0.5, size=n2)
    y2 = a * x2 + b + np.random.normal(0.0, s, size=n2)
    
    # Generate abnormal data points (outliers)
    x1 = np.random.normal(0.5, 0.1, size=n1)
    y1 = a * x1 + b * 3 + np.abs(np.random.normal(0.0, s, size=n1))
    
    x = np.hstack([x2, x1]).reshape(-1,1)
    y = np.hstack([y2, y1])
    
    return x, y

x, y = reg_data_outlier(a=0.5, b=0.3, n=1000, s=0.2, outlier_rate=0.2)

# 1. OLS
model = LinearRegression()
result = model.fit(x.reshape(-1,1), y)

# Visualize the data and regression line
w = result.coef_
b = result.intercept_
y_hat = np.dot(w, x.T) + b

plt.figure(figsize=(6,5))
plt.scatter(x, y, s=5, c='r')
plt.plot(x, y_hat, c='blue')
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axhline(y=0, ls='--', lw=0.5, c='black')
plt.show()

print('\nOLS results:')
print('Regression line: y = {:.3f}x + {:.3f}'.format(w[0], b))
print('R2 score = {:.3f}'.format(r2_score(y, y_hat)))

# RANSAC
n_sample =10     # the number of samples chosen randomly from original data
z_prob = 0.99    # the probability z
w_prob = 0.8     # the probability w

# The maximum number of attempts to find a consensus set
k_maxiter = int(np.log(1.0 - z_prob) / np.log(1.0 - w_prob ** n_sample))

# RANSACRegressor/residual_threshold:
# the threshold is chosen as the MAD (median absolute deviation) of the 
# target values y
threshold = np.median(np.abs(y - np.median(y)))

ransac_w = 0   # slope
ransac_b = 0   # intercept
ransac_c = 0   # count within the error tolerance
for i in range(k_maxiter):
    # sampling without replacement
    idx = np.random.choice(np.arange(0, x.shape[0]-1), n_sample, replace=False)
    xs = x[idx]
    ys = y[idx]
    
    # OLS Regression
    model = LinearRegression()
    result = model.fit(xs, ys)
    
    # Calculate the absolute value of residuals.
    y_pred = np.dot(result.coef_, x.T) + result.intercept_
    residual = np.abs(y - y_pred)
    
    # Count the number of times the residual is less than the threshold.
    count = (residual < threshold).sum()
    
    # Find the regression line where the count is largest.
    if count > ransac_c:
        ransac_c = count
        ransac_w = result.coef_
        ransac_b = result.intercept_

y_pred = np.dot(ransac_w, x.T) + ransac_b

# Visually check the data and final regression line
plt.figure(figsize=(6,5))
plt.scatter(x, y, s=5, c='r')
plt.plot(x, y_pred, c='blue')
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axhline(y=0, ls='--', lw=0.5, c='black')
plt.show()

print('\nRANSAC results:')
print('The maximum number of k = {}'.format(k_maxiter))
print('Threshold = {:.3f}'.format(threshold))
print('Regression line: y = {:.3f}x + {:.3f}'.format(ransac_w[0], ransac_b))
print('R2 score = {:.3f}'.format(r2_score(y, y_pred)))
