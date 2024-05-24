# [MXML-9-03] 3.AdaBoost(multiclass).py
# [1] Yoav Freund et, al., 1999, A Short Introduction to Boosting
# [2] Ji Zhu, et, al., 2006, Multi-class AdaBoost
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/tPeRalG7gYY
# 
import numpy as np
import random as rd
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
      
# Create training data. y = {0, 1, 2}
x, y = make_blobs(n_samples=300, n_features=2, 
                centers=[[0., 0.], [0.25, 0.5], [0.5, 0.]], 
                cluster_std=0.18, center_box=(-1., 1.))

K = np.unique(y).shape[0]  # the number of target class = 3
m = x.shape[0]
R = np.arange(m)
T = 50

# [2] Algorithm 2: SAMME
# SAMME : Stagewise Additive Modeling using a Multi-class Exponential loss function
weights = [np.array(np.ones(shape=(m,)) / m)]
eps = []      # epsilon history
alphas = []   # alpha history
models = []   # base learner models
for t in range(T):
    # sampling according to the weights
    s_idx = np.array(rd.choices(R, weights=weights[-1], k=m))
    sx = x[s_idx]      # sample x
    sy = y[s_idx]      # sample y

    # Make sure to select at least one class of y.
    # Among y=0, 1, and 2, if only the data points with y=1 and 2 
    # are selected, select one more data point with y=0.
    uy = np.unique(sy)
    if uy.shape[0] < K:
        unseen = list(set(np.unique(y)) - set(uy))
        for u in unseen:
            ui = rd.choices(np.where(y_train == u)[0], k=1)
            sx = np.vstack([sx, x_train[ui]])
            sy = np.hstack([sy, y_train[ui]])
    
    # base weak learner
    model = DecisionTreeClassifier(max_depth=1)
    model.fit(sx, sy)  # fit the model to sample data
    
    # calculate error (epsilon)
    y_pred = model.predict(x)   # predict entire training data
    i_not = np.array(y_pred != y).astype(int)  # I(y_pred ≠ y)
    eps.append(np.sum(weights[-1] * i_not))

    # calculate alpha using the error
    # For alpha to be positive, eps must be less than 1 - 1/K.
    # If eps is greater than 1 - 1/K, it means it is worse than a 
    # random prediction. If so, initialize the weights to 1/m again.
    if eps[-1] > 1 - 1/K:
        weights.append(np.array(np.ones(shape=(m,)) / m))
        alphas.append(0.0)
        print('weight re-initialized at t =', t)
    else:
        alpha = np.log((1 - eps[-1]) / (eps[-1] + 1e-8)) + np.log(K - 1)
        alphas.append(alpha)
        
        new_weights = weights[-1] * np.exp(alpha * i_not)
        weights.append(new_weights / new_weights.sum())  # normalize
    
    models.append(model)

# prediction
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))
H = np.zeros(shape=(x_test.shape[0], K))
for t in range(T):
    h = models[t].predict(x_test)
    oh = np.eye(K)[h]
    H += alphas[t] * oh

y_pred = np.argmax(H, axis=1)

# visualize training data and the sampling weights
def plot_train(x, y, w):
    plt.figure(figsize=(5,5))
    color = [['red', 'blue', 'green'][a] for a in y]
    plt.scatter(x[:, 0], x[:, 1], s=w*10000, c=color, alpha=0.5)
    plt.xlim(-0.5, 1.0)    
    plt.ylim(-0.5, 1.0)
    plt.show()
    
# visualize decision boundary
def plot_boundary(x, y, x_test, y_pred):
    plt.figure(figsize=(5,5))
    color = [['red', 'blue', 'green'][a] for a in y_pred]
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
# Check that ε are all less than 1 - 1/K 
# and that α and ε are inversely proportional.
plt.plot(eps, marker='o', markersize=4, c='red', lw=1, label='epsilon')
plt.legend()
plt.show()

plt.plot(alphas, marker='o', markersize=4, c='blue', lw=1, label='alpha')
plt.legend()
plt.show()


