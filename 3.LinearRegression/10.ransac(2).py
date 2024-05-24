# [MXML-3-07] 10.ransac(2).py
# Implementing RANSAC using sklearn's RANSACRegressor.
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
from sklearn.linear_model import LinearRegression, RANSACRegressor
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

# min_samples:
#    min_samples is chosen as X.shape[1] + 1.
# stop_probability:
#    RANSAC iteration stops if at least one outlier-free set of the training 
#    data is sampled in RANSAC. This requires to generate at least N samples 
#    (iterations):
# residual_threshold:
#    By default the threshold is chosen as the MAD (median absolute deviation) 
#    of the target values y.
model = RANSACRegressor(LinearRegression(),
                        stop_probability = 0.99,     # default
                        residual_threshold = None,   # default
                        min_samples = 10)

model.fit(x, y)

w = model.estimator_.coef_
b = model.estimator_.intercept_

# Visually check the data and final regression line
y_pred = model.predict(x)
plt.figure(figsize=(6,5))
plt.scatter(x, y, s=5, c='r')
plt.plot(x, y_pred, c='blue')
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axhline(y=0, ls='--', lw=0.5, c='black')
plt.show()

print('\nRANSAC results:')
print('Regression line: y = {:.3f}x + {:.3f}'.format(w[0], b))
print('R2 score = {:.3f}'.format(r2_score(y, y_pred)))


