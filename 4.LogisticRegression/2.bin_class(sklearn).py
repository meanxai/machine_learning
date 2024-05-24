# [MXML-4-02] 2.bin_class(sklearn).py
# Logistic Regression : binary classification
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/MifHxwJYOyU
#
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Create simple training data
x, y = make_blobs(n_samples=1000, n_features=2, 
                centers=[[1., 1.], [2., 2.]], 
                cluster_std=0.5)
                
# Visually check the data
color = ['red', 'blue']
for i in [0, 1]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c=color[i], s = 10, 
                alpha = 0.5, label='class-'+str(i))
plt.legend()
plt.show()                

# Split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Create a model and fit it to training data.
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict the classes of test data, and measure the accuracy.
y_pred = model.predict(x_test)
acc = (y_pred == y_test).mean()
print('\nAccuracy of test data = {:.3f}'.format(acc))

# Visually check the decision boundary.
# reference : 
# https://psrivasin.medium.com/
#   plotting-decision-boundaries-using-numpy-and-matplotlib-f5613d8acd19
x1_min, x1_max = x_test[:, 0].min() - 0.1, x_test[:,0].max() + 0.1
y1_min, y1_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                     np.linspace(y1_min, y1_max, 100))
x_in = np.c_[x1.ravel(), x2.ravel()]  # shape = (10000, 2)

# Predict all the data points in the meshgrid area.
y_pred = model.predict(x_in)

# Drawing the data and decision boundary
y_pred = y_pred.reshape(x1.shape)  # shape = (100, 100)

plt.figure(figsize=(5,5))
m = ['o', '^']
color = ['red', 'blue']
for i in [0, 1]:
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], 
                c=color[i], 
                marker = m[i],
                s = 40,
                edgecolor = 'black',
                alpha = 0.5,
                label='class-'+str(i))
plt.contour(x1, x2, y_pred, cmap=ListedColormap(['red', 'blue']), alpha=0.5)

plt.axis('tight')
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
