# [MXML-9-02] 2.AdaBoost(binary2).py
# Using y = {0, 1} instead of y = {-1, +1}
#
# [1] Yoav Freund et, al., 1999, A Short Introduction to Boosting
# [2] Ji Zhu, et, al., 2006, Multi-class AdaBoost
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/LVStXzGpA7Y
# 
import numpy as np
import random as rd
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
      
# Create training data
x, y = make_blobs(n_samples=200, n_features=2, 
                centers=[[0., 0.], [0.5, 0.5]], 
                cluster_std=0.2, center_box=(-1., 1.))

m = x.shape[0]
R = np.arange(m)
T = 50

weights = [np.array(np.ones(shape=(m,)) / m)]
eps = []      # epsilon history
alphas = []   # alpha history
models = []   # base learner models
for t in range(T):
    s_idx = np.array(rd.choices(R, weights=weights[-1], k=m))  # weighted sample index
    sx = x[s_idx]      # sample x
    sy = y[s_idx]      # sample y
    
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(sx, sy)  # fit the model to sample data
    
    y_pred = model.predict(x)  # predict entire training data
    i_not = np.array(y_pred != y).astype(int)  # I(y_pred ≠ y)
    eps.append(np.sum(weights[-1] * i_not))

    if eps[-1] > 0.5:
        weights.append(np.array(np.ones(shape=(m,)) / m))
        alphas.append(0.0)
        print('weight re-initialized at t =', t)
    else:
        alpha = 0.5 * np.log((1 - eps[-1]) / (eps[-1] + 1e-8))
        alphas.append(alpha)
        
        new_weights = weights[-1] * np.exp(alpha * i_not)
        weights.append(new_weights / new_weights.sum())  # normalize
    
    models.append(model)

H = np.zeros(shape=(5, 2))
h = np.array([1,0,1,0,1])
    
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))
H = np.zeros(shape=(x_test.shape[0], 2))
for t in range(T):
    h = models[t].predict(x_test)
    oh = np.eye(2)[h]       # one-hot encoding
    H += alphas[t] * oh

y_pred = np.argmax(H, axis=1)

# visualize training data and the sampling weights
def plot_train(x, y, w):
    plt.figure(figsize=(5,5))
    color = ['red' if a == 1 else 'blue' for a in y]
    plt.scatter(x[:, 0], x[:, 1], s=w*10000, c=color, alpha=0.5)
    plt.xlim(-0.5, 1.0)    
    plt.ylim(-0.5, 1.0)
    plt.show()
    
# visualize decision boundary
def plot_boundary(x, y, x_test, y_pred):
    plt.figure(figsize=(5,5))
    color = ['red' if a == 1 else 'blue' for a in y_pred]
    plt.scatter(x_test[:, 0], x_test[:, 1], s=100, c=color, alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], s=80, c='black')
    plt.scatter(x[:, 0], x[:, 1], s=10, c='yellow')
    plt.xlim(-0.5, 1.0)
    plt.ylim(-0.5, 1.0)
    plt.show()

plot_train(x, y, w=np.array(np.ones(shape=(m,)) / m))
plot_train(x, y, w=weights[-1])
plot_boundary(x, y, x_test, y_pred)

# Check the changes in α (alpha), ε (eps).
# Check that ε are all less than 0.5 and that α and ε are inversely proportional.
plt.plot(eps, marker='o', markersize=4, c='red', lw=1, label='epsilon')
plt.plot(alphas, marker='o', markersize=4, c='blue', lw=1, label='alpha')
plt.legend()
plt.show()


