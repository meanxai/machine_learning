# [MXML-4-05] 11.lwlr_2(sklearn).py
# Check the non-linear decision boundary
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
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate a simple dataset
def lwlr_data2(n, s):
    n1 = int(n /  3)
    x, y = [], []
    for a, b, c, m in [(1, 1, 0, n1), 
                       (2, 2, 1, n-n1*2), (3, 1, 0, n1)]:
        x1 = np.random.normal(a, s, m).reshape(-1,1)
        x2 = np.random.normal(b, s, m).reshape(-1,1)
        x.extend(np.hstack([x1, x2]))
        y.extend(np.ones(m) * c)
    x = np.array(x).reshape(-1, 2)
    y = np.array(y).astype('int8').reshape(-1, 1)
    return x, y.reshape(-1,)
x, y = lwlr_data2(n=1000, s=0.5)

# Visually check the data distribution.
m = ['o', '^']
color = ['red', 'blue']
plt.figure(figsize=(5,5))
for i in [0, 1]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], 
                c=color[i], 
                marker = m[i],
                s = 20,
                edgecolor = 'black',
                alpha = 0.5,
                label='class-'+str(i))
plt.legend()
plt.show()

# Split the data into the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Calculating the weights of training data points
# xx : training data, xx : test data
def get_weight(xx, tx, tau):
    distance = np.sum(np.square(xx - tx), axis=1)
    w = np.exp(-distance / (2 * tau * tau))
    return w

# Predict the classes of the test data
y_prob = []
tau = 0.1
for tx in x_test:
    weight = get_weight(x_train, tx, tau)
    model = LogisticRegression()
    model.fit(x_train, y_train, sample_weight = weight)
    y_prob.append(model.predict_proba(tx.reshape(-1, 2))[:, 1])
y_prob = np.array(y_prob).reshape(-1,)

# Measure the accuracy of the test data
y_pred = (y_prob > 0.5).astype('int8')
acc = (y_pred == y_test).mean()
print('\nAccuracy of the test data = {:.3f}'.format(acc))

# Visualize the non-linear decision boundary
# reference : 
# https://psrivasin.medium.com/
#   plotting-decision-boundaries-using-numpy-and-matplotlib-f5613d8acd19    
x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:,0].max() + 0.1
y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))
x_in = np.c_[xx.ravel(), yy.ravel()]

# Predict the classes of the data points in the x_in variable.
y_prob = []
for tx in x_in:
    weight = get_weight(x_train, tx, tau)
    
    model = LogisticRegression()
    model.fit(x_train, y_train, sample_weight = weight)
    y_prob.append(model.predict_proba(tx.reshape(-1, 2))[:, 1])
y_prob = np.array(y_prob).reshape(-1,)
y_pred = (y_prob > 0.5).astype('int8')

# Draw the decision boundary
y_pred = np.round(y_pred).reshape(xx.shape)

plt.figure(figsize=(5, 5))
for i in [0, 1]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], 
                c=color[i], 
                marker = m[i],
                s = 40,
                edgecolor = 'black',
                alpha = 0.5,
                label='class-' + str(i))
plt.contour(xx, yy, y_pred, cmap=ListedColormap(['red', 'blue']), alpha=0.5)
plt.axis('tight')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
