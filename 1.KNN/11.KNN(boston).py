# [MXML-1-07] 11.KNN(boston).py
# Predict the house prices in Boston using KNN
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/_ZxTTvbZOtc
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pickle

# Read Boston house price dataset
with open('data/boston_house.pkl', 'rb') as f:
    data = pickle.load(f)
x = data['data']      # shape = (506, 13)
y = data['target']    # shape = (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Z-score Normalization
x_mu = x_train.mean(axis=0)
x_sd = x_train.std(axis=0)
y_mu = y_train.mean()
y_sd = y_train.std()
zx_train = (x_train - x_mu) / x_sd
zy_train = (y_train - y_mu) / y_sd
zx_test = (x_test - x_mu) / x_sd
zy_test = (y_test - y_mu) / y_sd

# Visually check the actual and predicted prices
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(5, 4))
    plt.scatter(y_true, y_pred, s=20, c='r')
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.show()
    
# Simple average method
model1 = KNeighborsRegressor(n_neighbors = 10)
model1.fit(zx_train, zy_train)
y_pred1 = model1.predict(zx_test) * y_sd + y_mu
plot_predictions(y_test, y_pred1)
print('KNN R2 = {:.3f}'.format(model1.score(zx_test, zy_test)))

# Weighted average method
model2 = KNeighborsRegressor(n_neighbors = 30, weights='distance')
model2.fit(zx_train, zy_train)
y_pred2 = model2.predict(zx_test) * y_sd + y_mu
plot_predictions(y_test, y_pred2)
print('WKNN R2 = {:.3f}'.format(model2.score(zx_test, zy_test)))

a=np.array([.751, .671, .797, .802, .737, .789, .771, .735, .736, .668])
a=np.array([.741, .669, .757, .764, .657, .703, .718, .747, .682, .647])
a.mean()
