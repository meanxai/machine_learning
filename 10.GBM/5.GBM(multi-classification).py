# [MXML-10-07] 5.GBM(multi-classification).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/qfdkyWOqRPg
# 
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Create training data
x, y = make_blobs(n_samples=300, n_features=2, 
                  centers=[[0., 0.], [0.5, 0.5], [1.0, 0.]], 
                  cluster_std=0.18, center_box=(-1., 1.))

# One-hot encoding of y
y_ohe = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()

# log(odds) --> probability
def F2P(f):
    return 1. / (1. + np.exp(-f))

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

n_data = x.shape[0]
n_class = y_ohe.shape[1]
n_depth = 2    # tree depth
n_tree = 30    # the number of trees (M)
alpha = 0.1    # learning rate
u_class = np.unique(y)

# GradientBoostingClassifier: multi-class
# https://scikit-learn.org/stable/modules/ensemble.html
# Note: Classification with more than 2 classes requires the induction 
# of n_classes regression trees at each iteration, thus, the total number 
# of induced trees equals n_classes * n_estimators.

# step-1: Initialize model with a constant value.
# F0, Fm.shape = (n_data, n_class)
F0 = [np.log(y_ohe[:,c].mean()/(1. - y_ohe[:,c].mean())) for c in u_class]
Fm = np.tile(np.array(F0), [n_data, 1])

# Repeat
gb_models = []
loss = []
for m in range(n_tree):
    # step-2 (A): Compute so-called pseudo-residuals
    # y_hat, residual.shape = (n_data, n_class)
    prob = F2P(Fm)
    y_hat = softmax(prob)
    residual = y_ohe - y_hat
    
    # step-2 (B): Fit a regression tree to the residual
    gb_model = []
    for ci in u_class:
        gb_model.append(DecisionTreeRegressor(max_depth=n_depth))
        gb_model[-1].fit(x, residual[:, ci])
    
        # step-2 (C): compute gamma
        # leaf_id = The leaf node number to which x belongs.
        leaf_id = gb_model[-1].tree_.apply(x.astype(np.float32))
        
        # Replace the leaf values ​​of all leaf nodes with their gamma 
        # values, ​​and update Fm. 
        for j in np.unique(leaf_id):
            # xi=index of data points belonging to leaf node j.
            xi = np.where(leaf_id == j)[0]
            gamma = residual[:, ci][xi].sum() / \
                    (y_hat[:, ci][xi] * (1. - y_hat[:, ci][xi])).sum()
            
            # step-2 (D): Update the model
            Fm[:, ci][xi] += alpha * gamma
            
            # Replace the leaf values ​​with their gamma 
            gb_model[-1].tree_.value[j, 0, 0] = gamma
        
    gb_models.append(gb_model)
    
    # Calculating loss. loss = cross entropy.
    loss.append(-np.sum((y_ohe * np.log(y_hat + 1e-8)).sum(axis=1)))

# Check the loss history visually.
plt.figure(figsize=(5,4))
plt.plot(loss, c='red')
plt.xlabel('m : iteration')
plt.ylabel('loss: cross entropy')
plt.title('loss history')
plt.show()

# step-3: Output Fm(x) - Prediction of test data
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))

Fm = np.tile(np.array(F0), [x_test.shape[0], 1])
for model in gb_models:
    for ci in u_class:
        Fm[:, ci] += alpha * model[ci].predict(x_test)
        
y_prob = F2P(Fm)
y_soft = softmax(y_prob)
y_pred = np.argmax(y_soft, axis=1)  

# Visualize training and prediction results.
def plot_prediction(x, y, x_test, y_pred):
    plt.figure(figsize=(5,5))
    color = [['red', 'blue', 'green'][a] for a in y_pred]
    plt.scatter(x_test[:, 0], x_test[:, 1], s=100, c=color, 
                alpha=0.3)
    plt.scatter(x[:, 0], x[:, 1], s=80, c='black')
    plt.scatter(x[:, 0], x[:, 1], s=10, c='yellow')
    plt.xlim(-0.5, 1.5)    
    plt.ylim(-0.5, 1.0)
    plt.show()
    
# Visualize test data and y_pred.
plot_prediction(x, y, x_test, y_pred)

# Compare with Sklearn's GradientBoostingClassifier result.
from sklearn.ensemble import GradientBoostingClassifier

sk_model = GradientBoostingClassifier(n_estimators=n_tree, 
                                      learning_rate=alpha,
                                      max_depth=n_depth,
                                      criterion='squared_error')
sk_model.fit(x, y)

# Predict the target class of test data.
y_pred1 = sk_model.predict(x_test)

# Visualize test data and y_pred1.
plot_prediction(x, y, x_test, y_pred1)

sk_model.estimators_.shape
sk_model.estimators_[0]
