# [MXML-6-10] 12.cvxopt(svr_nonlinear).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/PzT4Cgz2HJE
# 
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

# Gaussian kernel function.
def kernel(a, b, gamma = 0.1):
    n = a.shape[0]
    m = b.shape[0]
    k = np.array([np.exp(-gamma * np.linalg.norm(a[i] - b[j])**2) 
                  for i in range(n) 
                  for j in range(m)]).reshape(n, m)
    return k

# Generate sinusoidal data with Gaussian noise added.
def nonlinear_data(n, s):
   rtn_x = []
   rtn_y = []
   for i in range(n):
       x = np.random.random()
       y = np.sin(2.0 * np.pi * x) + np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
       
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y).reshape(-1,1)

x, y = nonlinear_data(n=200, s=0.7)

eps = 2.0     # error tolerance range
gamma = 5.0   # gamma for RBF kernel
C = 1.0       # regularization constant
n = x.shape[0]

# Construct matrices required for QP in standard form.
K = kernel(x, x, gamma)
P = np.hstack([K, -K])
P = np.vstack([P, -P])
q = np.array([[eps]]) + np.vstack([-y, y])
A = np.array([1.] * n)
A = np.hstack([A, -A])
b = np.zeros((1,1))
G = np.vstack([-np.eye(2*n), np.eye(2*n)])
h = np.hstack([np.zeros(2*n), np.ones(2*n) * C])

P = cvxopt_matrix(P)
q = cvxopt_matrix(q)
A = cvxopt_matrix(A.reshape(1, -1))
b = cvxopt_matrix(b)
G = cvxopt_matrix(G)
h = cvxopt_matrix(h)

# solver parameters
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Perform QP
sol = cvxopt_solvers.qp(P, q, G, h, A, b)

# The lambdas, solution to the dual problem
lamb = np.array(sol['x'])

lamb_i = lamb[:n]    # λ
lamb_s = lamb[n:]    # λ*

# calculating the intercept, b
# [1] Alex Smola, et, al. 1998, A tutorial on support vector
#     regression. 1.4 Computing b, equation - (13)
# [2] Alex Smola, et, al. 2004, A tutorial on support vector
#     regression. 1.4 computing b, equation - (16)

# Calculate b
b = []

# 0 < λ < C
idx_i = np.logical_and(lamb_i > 1e-10, lamb_i < C - 1e-10)
if idx_i.shape[0] > 0:
    wx = np.sum((lamb_i - lamb_s) * kernel(x, x[idx_i], gamma), axis=0)
    b.extend((y[idx_i] - wx - eps).flatten())

# 0 < λ* < C
idx_s = np.logical_and(lamb_s > 1e-10, lamb_s < C - 1e-10)
if idx_s.shape[0] > 0:
    wx = np.sum((lamb_i - lamb_s) * kernel(x, x[idx_s], gamma), axis=0)
    b.extend((y[idx_s] - wx + eps).flatten())

if len(b) > 0:
   b = np.mean(b)
   i_sup = np.where(lamb_i > 1e-5)[0]
   s_sup = np.where(lamb_s > 1e-5)[0]
   
   # Visualize the data and decision function
   plt.figure(figsize=(6,5))
   plt.scatter(x, y, c='blue', alpha=0.5)
   plt.scatter(x[i_sup], y[i_sup], c='blue', ec='red', lw=2.0, alpha=0.5)
   plt.scatter(x[s_sup], y[s_sup], c='blue', ec='red', lw=2.0, alpha=0.5)
   x_dec = np.linspace(0, 1, 50).reshape(-1, 1)
   
   # decision function
   wx = np.sum((lamb_i - lamb_s) * kernel(x, x_dec, gamma), axis=0).reshape(-1, 1)
   y_dec = wx + b
   
   plt.plot(x_dec, y_dec, c='red', lw=2.0)
   plt.plot(x_dec, y_dec + eps, '--', c='orange', lw=1.0)
   plt.plot(x_dec, y_dec - eps, '--', c='orange', lw=1.0)
   plt.xlim(0, 1)
   plt.ylim(0, 7)
   plt.title('cvxopt version : b=' + str(b.round(4)))
   plt.show()
else:
   print('Failed to calculate b.')   

# Compare with the results of sklearn.svm.SVR
from sklearn.svm import SVR
model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=eps, )
model.fit(x, y.flatten())

b1 = model.intercept_
lamb1 = model.dual_coef_
sv_idx = model.support_
sv_x = x[sv_idx]
sv_y = y[sv_idx]

# Visualize the data and decision function
plt.figure(figsize=(6,5))
plt.scatter(x, y, c='blue', alpha=0.5)
plt.scatter(sv_x, sv_y, c='blue', ec='red', lw=2.0, alpha=0.5)
x_dec = np.linspace(0, 1, 50).reshape(-1, 1)
y_dec = model.predict(x_dec)
plt.plot(x_dec, y_dec, c='red', lw=2.0)
plt.plot(x_dec, y_dec + eps, '--', c='orange', lw=1.0)
plt.plot(x_dec, y_dec - eps, '--', c='orange', lw=1.0)
plt.xlim(0, 1)
plt.ylim(0, 7)
plt.title('sklearn.svm.SVR version : b='+str(b1[0].round(4)))
plt.show()
