# [MXML-6-09] 11.cvxopt(svr_linear).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/8VFw2DFeJE4
# 
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

# Create a dataset. y = ax + b + noise
def linear_data(a, b, n, s):
   rtn_x = []
   rtn_y = []
   for i in range(n):
       x = np.random.normal(0.0, 0.5)
       y = a * x + b + np.random.normal(0.0, s)
       rtn_x.append(x)
       rtn_y.append(y)
       
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y).reshape(-1,1)

x, y = linear_data(a=0.5, b=0.3, n=200, s=0.2)

eps = 0.2
C = 2.0
n = x.shape[0]

# Construct matrices required for QP in standard form.
K = np.dot(x, x.T)
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

# Calculate w using the lambdas, which is the solution to QP.
lamb_i = lamb[:n]    # λ
lamb_s = lamb[n:]    # λ*
w = np.sum((lamb_i - lamb_s) * x, axis=0)

# calculating the intercept, b
# [1] Alex Smola, et, al. 1998, A tutorial on support vector
#     regression. 1.4 Computing b, equation - (13)
# [2] Alex Smola, et, al. 2004, A tutorial on support vector
#     regression. 1.4 computing b, equation - (16)

# Calculate b
b = []
# 0 < λ < C
idx_i = np.logical_and(lamb_i > 1e-5, lamb_i < C - 1e-5)
if idx_i.shape[0] > 0:
    b.extend((y[idx_i] - w * x[idx_i] - eps).flatten())

# 0 < λ* < C
idx_s = np.logical_and(lamb_s > 1e-5, lamb_s < C - 1e-5)
if idx_s.shape[0] > 0:
    b.extend((y[idx_s] - w * x[idx_s] + eps).flatten())

if len(b) > 0:
   b = np.mean(b)
   i_sup = np.where(lamb_i > 1e-5)[0]
   s_sup = np.where(lamb_s > 1e-5)[0]
   
   # Visualize the data and decision function
   plt.figure(figsize=(6,6))
   plt.scatter(x, y, c='blue', alpha=0.5)
   plt.scatter(x[i_sup], y[i_sup], c='blue', ec='red', lw=2.0, alpha=0.5)
   plt.scatter(x[s_sup], y[s_sup], c='blue', ec='red', lw=2.0, alpha=0.5)
   x_dec = np.linspace(-1, 1.5, 50).reshape(-1, 1)
   y_dec = w * x_dec + b
   plt.plot(x_dec, y_dec, c='red', lw=1.0)
   plt.plot(x_dec, y_dec + eps, '--', c='orange', lw=1.0)
   plt.plot(x_dec, y_dec - eps, '--', c='orange', lw=1.0)
   plt.xlim(-1, 1.5)
   plt.ylim(-0.5, 1.5)
   plt.title('cvxopt version : w='+str(w[0].round(4))+', b='+str(b.round(4)))
   plt.show()
else:
   print('Failed to calculate b.')
    
# Compare with the results of sklearn.svm.SVR
from sklearn.svm import SVR
model = SVR(C=C, epsilon=eps, kernel='linear')
model.fit(x, y.flatten())

w1 = model.coef_
b1 = model.intercept_
lamb1 = model.dual_coef_
sv_idx = model.support_
sv_x = x[sv_idx]
sv_y = y[sv_idx]

# Visualize the data and decision function
plt.figure(figsize=(6,6))
plt.scatter(x, y, c='blue', alpha=0.5)
plt.scatter(sv_x, sv_y, c='blue', ec='red', lw=2.0, alpha=0.5)
x_dec = np.linspace(-1, 1.5, 50).reshape(-1, 1)
y_dec = w1 * x_dec + b1
plt.plot(x_dec, y_dec, c='red', lw=1.0)
plt.plot(x_dec, y_dec + eps, '--', c='orange', lw=1.0)
plt.plot(x_dec, y_dec - eps, '--', c='orange', lw=1.0)
plt.xlim(-1, 1.5)
plt.ylim(-0.5, 1.5)
plt.title('sklearn.svm.SVR version : w='+str(w1[0][0].round(4))+', b='+str(b1[0].round(4)))
plt.show()
