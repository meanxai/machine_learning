# [MXML-3-07] 11.boston(ransac).py
# Predict the Boston house prices using RANSAC
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
from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.model_selection import train_test_split
import pickle

# Read Boston house price dataset
with open('data/boston_house.pkl', 'rb') as f:
    data = pickle.load(f)
x = data['data']      # shape = (506, 13)
y = data['target']    # shape = (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# min_samples:
#    min_samples is chosen as X.shape[1] + 1.
# stop_probability:
#    RANSAC iteration stops if at least one outlier-free set of the training 
#    data is sampled in RANSAC. This requires to generate at least N samples 
#    (iterations):
# residual_threshold:
#    By default the threshold is chosen as the MAD (median absolute deviation) 
#    of the target values y.
model = RANSACRegressor(Ridge(alpha=0.01),
                        stop_probability = 0.99,     # default
                        residual_threshold = None,   # default
                        min_samples = 50)

model.fit(x_train, y_train)

# Visually check the actual and predicted prices
y_pred = model.predict(x_test)
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, s=20, c='r')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

print('RANSAC R2 = {:.3f}'.format(model.score(x_test, y_test)))

