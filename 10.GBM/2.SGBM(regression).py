# [MXML-10-03] 2.SGBM(regression).py 
# Stochastic Gradient Boosting Method (1999, Friedman)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/hF-1HHKPxq4
# 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create training data for regression
def nonlinear_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.random()
       y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
       
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)

# Visualize the training data and prediction results
def plot_prediction(x, y, x_test, y_pred):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, c='blue', s=20, alpha=0.5, label='train data')
    plt.plot(x_test, y_pred, c='red', lw=2.0, label='prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 7)
    plt.legend()
    plt.show()

# Create training data
x, y = nonlinear_data(n=500, s=0.5)

n_data = x.shape[0]
n_depth = 3         # tree depth (weak learner)
n_tree = 50         # the number of trees (M)
f_rate = 0.5        # rate of sampling
lr = 0.05           # learning rate

# step-1: Initialize model with a constant value.
F0 = y.mean()

# Training
Fm = np.repeat(F0, n_data)
models = []
loss = []
for m in range(n_tree):
    # data sampling without replacement
    si = np.random.choice(range(n_data), int(n_data * f_rate), replace=False)
    
    # step-2 (A): Compute so-called pseudo-residuals
    residual = y[si] - Fm[si]
    
    # step-2 (B): Fit a regression tree to the residual
    gb_model = DecisionTreeRegressor(max_depth=n_depth)
    gb_model.fit(x[si], residual)
    
    # step-2 (C): compute gamma (prediction)
    gamma = gb_model.predict(x)
    
    # step-2 (D): Update the model
    Fm = Fm + lr * gamma
    
    # Store trained tree models
    models.append(gb_model)
    
    # Calculate loss. loss = mean squared error.
    loss.append(((y - Fm) ** 2).sum())

# Check the loss history
plt.figure(figsize=(6,4))
plt.plot(loss, c='red')
plt.xlabel('m : iteration')
plt.ylabel('loss: mean squared error')
plt.title('loss history')
plt.show()

# step-3: Output Fm(x) - Prediction
y_pred = F0
x_test = np.linspace(0, 1, 50).reshape(-1, 1)
for model in models:
    y_pred += lr * model.predict(x_test)

# Visualize the training data and prediction results
plot_prediction(x, y, x_test, y_pred)

# Compare with the results of sklearn’s GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
sk_model = GradientBoostingRegressor(n_estimators=n_tree, 
                                     learning_rate=lr, 
                                     max_depth=n_depth,
                                     subsample=f_rate)

sk_model.fit(x, y)                 # Training
y_pred = sk_model.predict(x_test)  # Prediction

plot_prediction(x, y, x_test, y_pred)