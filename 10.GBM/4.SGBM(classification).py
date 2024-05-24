# [MXML-10-06] 4.SGBM(classification).py
# Stochastic Gradient Boosting Method (1999, Friedman)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/YZdoxpWe5ng
# 
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# log(odds) --> probability
def F2P(f):
    return 1. / (1. + np.exp(-f))

# Creating training data
x, y = make_blobs(n_samples=200, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.5]], 
                  cluster_std=0.2, center_box=(-1., 1.))

n_data = x.shape[0]
n_depth = 2    # tree depth
n_tree = 200   # the number of trees (M)
f_rate = 0.5   # sampling ratio
alpha = 0.05   # learning rate

# step-1: Initialize model with a constant value.
F0 = np.log(y.mean() / (1. - y.mean()))
Fm = np.repeat(F0, n_data)

# Repeat
models = []
loss = []
for m in range(n_tree):
    # data sampling without replacement
    si = np.random.choice(range(n_data), int(n_data * f_rate), replace=False)

    # step-2 (A): Compute so-called pseudo-residuals
    y_hat = F2P(Fm)
    residual = y[si] - y_hat[si]
    
    # step-2 (B): Fit a regression tree to the residual
    gb_model = DecisionTreeRegressor(max_depth=n_depth)
    gb_model.fit(x[si], residual)
    
    # The leaf nodes of this tree contain the average of the 
    # residuals. The predict() function returns this average value. 
    # We replace these values ​​with the leaf values gamma. Then the 
    # predict() function will return the gamma.
    
    # step-2 (C): compute gamma
    # leaf_id = The leaf node number to which x[si] belongs.
    leaf_id = gb_model.tree_.apply(x[si].astype(np.float32))
    
    # Replace the leaf values ​​of all leaf nodes with their gamma 
    # values, ​​and update Fm.
    for j in np.unique(leaf_id):
        # i=Index of data points belonging to leaf node j.
        i = np.where(leaf_id == j)[0]
        xi = si[i]
        gamma = residual[i].sum() / (y_hat[xi] * (1. - y_hat[xi])).sum()

        # step-2 (D): Update the model
        Fm[xi] += alpha * gamma
        
        # Replace the leaf values ​​with their gamma
        # gb_model.tree_.value.shape = (7, 1, 1)
        gb_model.tree_.value[j, 0, 0] = gamma

    # save the trained model
    models.append(gb_model)
    
    # Calculating loss. loss = binary cross entropy.
    loss.append(-(y * np.log(y_hat + 1e-8) + \
                 (1.- y) * np.log(1.- y_hat + 1e-8)).sum())

# Check the loss history visually.
plt.figure(figsize=(5,4))
plt.plot(loss, c='red')
plt.xlabel('m : iteration')
plt.ylabel('loss: binary cross entropy')
plt.title('loss history')
plt.show()

# step-3: Output Fm(x) - Prediction of test data
Fm = F0
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))

for model in models:
    Fm += alpha * model.predict(x_test)
    
y_prob = F2P(Fm)
y_pred = (y_prob > 0.5).astype('uint8')

# Visualize training and prediction results.
def plot_prediction(x, y, x_test, y_pred):
    plt.figure(figsize=(5,5))
    color = ['red' if a == 1 else 'blue' for a in y_pred]
    plt.scatter(x_test[:, 0], x_test[:, 1], s=100, c=color, 
                alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], s=80, c='black')
    plt.scatter(x[:, 0], x[:, 1], s=10, c='yellow')
    plt.xlim(-0.5, 1.0)    
    plt.ylim(-0.5, 1.0)
    plt.show()
    
# Visualize test data and y_pred.
plot_prediction(x, y, x_test, y_pred)

# Compare with Sklearn's GradientBoostingClassifier result.
from sklearn.ensemble import GradientBoostingClassifier
sk_model = GradientBoostingClassifier(n_estimators=n_tree, 
                                      learning_rate=alpha,
                                      max_depth=n_depth,
                                      subsample=f_rate)
sk_model.fit(x, y)

# Predict the target class of test data.
y_pred1 = sk_model.predict(x_test)

# Visualize test data and y_pred1.
plot_prediction(x, y, x_test, y_pred1)

