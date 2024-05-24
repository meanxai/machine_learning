# [MXML-11-06] 2.XGBoost(classification).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/oKLBon15bTc
# 
import numpy as np
from sklearn.datasets import make_blobs
from MyXGBoostClassifier import MyXGBClassifier
import matplotlib.pyplot as plt

# Plot the training and test data, and the prediction result
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
   
# Generate the training data
x, y = make_blobs(n_samples=200, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.5]], 
                  cluster_std=0.18, center_box=(-1., 1.))

# y_init = y.mean()     # initial prediction
y_init = np.repeat(y.mean(), y.shape[0])
n_depth = 3           # # tree depth
n_tree = 20           # the number of trees
eta = 0.3             # learning rate
reg_lambda = 0.1      # regularization constant
prune_gamma = 0.01    # pruning constant

my_model = MyXGBClassifier(n_estimators=n_tree,
                           max_depth=n_depth,
                           learning_rate=eta,
                           prune_gamma = prune_gamma,
                           reg_lambda=reg_lambda,
                           base_score = y_init)
loss = my_model.fit(x, y)

# Check the loss history
plt.figure(figsize=(5,4))
plt.plot(loss, c='red')
plt.xlabel('m : iteration')
plt.ylabel('loss: binary cross entropy')
plt.title('loss history')
plt.show()

x_test = np.random.uniform(-0.5, 1.5, (1000, 2))
y_pred = my_model.predict(x_test)

# Plot the training and test data, and the prediction result
plot_prediction(x, y, x_test, y_pred)

# Compare with the results from XGBRegressor library.
# https://xgboost.readthedocs.io/en/stable/python/python_api.html
#       #module-xgboost.sklearn
# ---------------------------------------------------------------
from xgboost import XGBClassifier

xg_model = XGBClassifier(objective='binary:logistic',
                         tree_method = 'exact',
                         n_estimators=n_tree,
                         max_depth=n_depth,
                         learning_rate=eta,
                         gamma=prune_gamma,
                         reg_lambda=reg_lambda,
                         base_score=y_init)

xg_model.fit(x, y)

# Predict the target class of the test data and visualize the result
y_pred = xg_model.predict(x_test)
plot_prediction(x, y, x_test, y_pred)

# plt.figure(figsize=(5,5))
# color = ['red' if a == 1 else 'blue' for a in y]
# plt.scatter(x[:, 0], x[:, 1], s=80, alpha=0.5, c=color)
# plt.xlim(-0.5, 1.0)    
# plt.ylim(-0.5, 1.0)
# plt.show()