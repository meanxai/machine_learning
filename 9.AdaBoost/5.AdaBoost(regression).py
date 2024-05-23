# [MXML-9-04] 5.AdaBoost(regression).py
# [1] Harris Drucker et, al., 1997, Improving Regressors using Boosting Techniques
import numpy as np
import random as rd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create training data
def noisy_sine_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.random()
       y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)
x, y = noisy_sine_data(n=500, s=0.5)
    
N = x.shape[0]
R = np.arange(N)
T = 100

weights = np.array(np.ones(shape=(N,)) / N)
beta = []     # beta history
models = []   # save base learners for prediction
for t in range(T):
    s_idx = np.array(rd.choices(R, weights=weights, k=N))
    sx = x[s_idx]      # sample x
    sy = y[s_idx]      # sample y
           
    # base learner
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(sx, sy)  # Fit the model to sample data
    
    # Calculate square loss
    y_pred = model.predict(x)       # predict entire training data   
    err = np.abs(y - y_pred)
    loss = (err / err.max()) ** 2      # squared loss
    
    loss_avg = np.sum(weights * loss)  # average loss
    if loss_avg > 0.5:
        print('stopped at t={}, loss_avg={:.2f}'.format(t, loss_avg))
        break
        
    # Calculate beta using average loss.
    beta.append(loss_avg / (1. - loss_avg))
    
    # Update weights using beta.
    new_weights = weights * np.power(beta[-1], (1. - loss))
    weights = new_weights / new_weights.sum()
    
    # save model
    models.append(model)

# Visualize training data and estimated curve
def plot_prediction(x, y, x_test, y_pred, title=""):
    plt.figure(figsize=(5, 3.5))
    plt.scatter(x, y, c='blue', s=20, alpha=0.5, label='train')
    plt.plot(x_test, y_pred, c='red', lw=2.0, label='prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 7)
    plt.legend()
    plt.title(title)
    plt.show()
    
# prediction.
n_test = 50
x_test = np.linspace(0, 1, n_test).reshape(-1, 1) # test data
log_beta = np.log(1. / np.array(beta))            # log(1/beta)
y_pred = np.array([m.predict(x_test) for m in models]).T

# Method-1: Using weighted average
w = log_beta/ log_beta.sum()    # normalize
wavg_pred = np.sum(y_pred * w, axis=1)
plot_prediction(x, y, x_test, wavg_pred, 'weighted average')

# weighted median: (sum of the lower w ≥ half of the total sum of w)
i_pred = np.argsort(y_pred, axis=1)
w_acc = np.cumsum(w[i_pred], axis=1)      # accumulated w
is_med = w_acc >= 0.5 * w_acc[:, -1][:, np.newaxis]
i_med = is_med.argmax(axis=1)             # 23
y_med = i_pred[np.arange(n_test), i_med]  # 34
wmed_pred = np.array(y_pred[np.arange(n_test), y_med])  # final estimate
plot_prediction(x, y, x_test, wmed_pred, 'weighted median')
   
# Let’s compare the results with sklearn’s AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

dt = DecisionTreeRegressor(max_depth=5)
model = AdaBoostRegressor(estimator=dt, n_estimators=T, loss='square')
model.fit(x, y)
sk_pred = model.predict(x_test)
plot_prediction(x, y, x_test, sk_pred, 'AdaBoostRegressor')