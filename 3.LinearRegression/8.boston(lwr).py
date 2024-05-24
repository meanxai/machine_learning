# [MXML-3-5] 8.bostn(lwr).py
# Predicting the Boston house price using LWR
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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle

# Read saved dataset
with open('data/boston_house.pkl', 'rb') as f:
    data = pickle.load(f)
x = data['data']      # shape = (506, 13)
y = data['target']    # shape = (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Find the weight for each data point.
# train: training data, test: test data point to be predicted
def get_weight(train, test, tau):
    d2 = np.sum(np.square(train - test), axis=1)
    w = np.exp(-d2 / (2. * tau * tau))
    return w

y_pred = []
for tx in x_test:
    weight = get_weight(x_train, tx, 50.0)
    model = Ridge(alpha=0.01)
    model.fit(x_train, y_train, sample_weight = weight)
    y_pred.append(model.predict(tx.reshape(1, -1))[0])

y_pred = np.array(y_pred).reshape(-1,)

# Visually check the actual and predicted y values ​​of the test data.
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, s=10, c='r')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

print('\nR2 (LWR) = {:.3f}'.format(r2_score(y_test, y_pred)))
