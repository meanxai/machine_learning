# [MXML-4-05] 10.lwlr(sklearn).py
# Use the sample_weight argument in sklearn's LogisticRegression model.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate a simple dataset
def lwlr_data1(n):
    n1 = int(n / 3)
    a = np.random.normal(-1.0, 0.5, n1)
    b = np.random.normal(1.0, 0.5, n1)
    c = np.random.normal(3.0, 0.5, n - n1 * 2)
    x = np.hstack([a, b, c]).reshape(-1, 1)
    y = np.hstack([np.zeros(n1), np.ones(n1), np.zeros(n - n1 * 2)])
    return x, y

# Generate training and test data
x, y = lwlr_data1(n=2000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Visualize the dataset
plt.figure(figsize=(6, 3))
plt.scatter(x_train, y_train, s=5, c='orange', label='train')
plt.scatter(x_test, y_test, marker='+', s=30, c='blue', label='test')
plt.legend()
plt.show()

# Calculating the weights of training data points
# xx : training data, xx : test data
def get_weight(xx, tx, tau):
    distance = np.sum(np.square(xx - tx), axis=1)
    w = np.exp(-distance / (2 * tau * tau))
    return w

y_prob = []
for tx in x_test:
    weight = get_weight(x_train, tx, 0.6)
    model = LogisticRegression()
    model.fit(x_train, y_train, sample_weight = weight)
    y_prob.append(model.predict_proba(tx.reshape(-1, 1))[:, 1])
    
y_prob = np.array(y_prob).reshape(-1,)

# Visually check the training and test data, and predicted probability.
plt.figure(figsize=(6, 3))
plt.scatter(x_train, y_train, s=5, c='orange', label='train')
plt.scatter(x_test, y_test, marker='+', s=30, c='blue', label='test')
plt.scatter(x_test, y_prob, s=5, c='red', label='prediction')
plt.legend()
plt.axhline(y=0.5, ls='--', lw=0.5, c='black')
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axvline(x=2, ls='--', lw=0.5, c='black')
plt.show()

# Measure the accuracy of the test data
y_pred = (y_prob > 0.5).astype('int8')
acc = (y_pred == y_test).mean()
print('\nAccuracy of the test data = {:.3f}'.format(acc))
