# [MXML-3-02] 1.scipy_opt(ols).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/YBk1FS1vmv4
#
from scipy import optimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# y = ax + b + Gaussian noise
def reg_data(a, b, n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.normal(0.0, 0.5)
       y = a * x + b + np.random.normal(0.0, s)
       rtn_x.append(x)
       rtn_y.append(y)    
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)
   
# Generate 1,000 data points drawn from y = ax + b + noise
# s : standard deviation of the noise distribution
x, y = reg_data(a=0.5, b=0.3, n=1000, s=0.2)

# y = w0 + w1*x1 + w2*x2 + ... → w0*x0 + w1*x1 + w2*x2 + ... (x0 = 1)
# y = [w0, w1, w2, ...] * [x0, x1, x2, ...].T  (T : transpose)
# y = W * X.T
X = np.hstack([np.ones([x.shape[0], 1]), x])
REG_CONST = 0.01   # regularization constant

# Loss function : Mean Squared Error
def ols_loss(W, args):
    e = np.dot(W, X.T) - y
    mse = np.mean(np.square(e))  # mean squared error
    loss = mse + REG_CONST * np.sum(np.square(W))
    
    # save W and loss
    if args[0] == True:
        trace_W.append([W, loss])
    return loss

# Perform optimization process
trace_W = []
result = optimize.minimize(ols_loss, [-4., 4], args=[True])
print(result)

# Plot the training data and draw the regression line.
y_hat = np.dot(result.x, X.T)
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=5, c='r')
plt.plot(x, y_hat, c='blue')
plt.axvline(x=0, ls='--', lw=0.5, c='black')
plt.axhline(y=0, ls='--', lw=0.5, c='black')
plt.show()

# Draw the loss function and the path to the optimal point.
m = 5
t = 0.1
w0, w1 = np.meshgrid(np.arange(-m, m, t), np.arange(-m, m, t))
zs = np.array([ols_loss([a,b], [False]) for [a, b] in zip(np.ravel(w0), np.ravel(w1))])
z = zs.reshape(w0.shape)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Draw the surface of the loss function
ax.plot_surface(w0, w1, z, alpha=0.7)

# Dwaw the path to the optimal point.
b = np.array([tw0 for [tw0, tw1], td in trace_W])
w = np.array([tw1 for [tw0, tw1], td in trace_W])
d = np.array([td for [tw0, tw1], td in trace_W])
ax.plot(b, w, d, marker='o', color="r")

ax.set_xlabel('W0 (bias)')
ax.set_ylabel('W1 (slope)')
ax.set_zlabel('distance')
ax.azim = -50
ax.elev = 50
plt.show()

# Check the R2 score
sst = np.sum(np.square(y - np.mean(y)))  # total sum of squares
sse = np.sum(np.square(y - y_hat))       # sum of squares of error
r2 = 1 - sse / sst
print('\nR2 score = {:.4f}'.format(r2))
print('R2 score = {:.4f}'.format(r2_score(y, y_hat)))
