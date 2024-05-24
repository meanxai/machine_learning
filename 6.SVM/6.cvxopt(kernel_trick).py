# [MXML-6-06] 6.cvxopt(kernel_trick).py
# Implemen nonlinear SVM using CVXOPT
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/-WVI6b19pag
# 
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

# 4 data samples. 2 ‘+’ samples, 2 ‘-’ samples
x = np.array([[0., 1.], [1., 1.], [1., 0.], [0., 0.]])
y = np.array([[-1.], [1.], [-1.], [1.]])

# kernel function
def kernel(a, b, p=3, r=0.5, type="rbf"):
    if k_type == "poly":
        return (1 + np.dot(a, b)) ** p
    else:
        return np.exp(-r * np.linalg.norm(a - b)**2)
    
C = 1.0          # regularization constant
N = x.shape[0]   # the number of data points
k_type = "poly"  # kernel type: poly or rbf

# Kernel matrix. k(xi, xj) = φ(xi)φ(xj).
K = np.array([kernel(x[i], x[j], type=k_type) 
        for i in range(N) 
        for j in range(N)]).reshape(N, N)
                  
# Construct the matrices required for QP in standard form.
H = np.outer(y, y) * K
P = cvxopt_matrix(H)
q = cvxopt_matrix(np.ones(N) * -1)
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

g = np.vstack([-np.eye(N), np.eye(N)])
G = cvxopt_matrix(g)

h1 = np.hstack([np.zeros(N), np.ones(N) * C])
h = cvxopt_matrix(h1)

# solver parameters
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Perform QP
sol = cvxopt_solvers.qp(P, q, G, h, A, b)

# the solution to the QP, λ
lamb = np.array(sol['x'])

# Find support vectors
sv_i = np.where(lamb > 1e-5)[0]
sv_m = lamb[sv_i]    # lambda
sv_x = x[sv_i]
sv_y = y[sv_i]

# Calculate b using the support vectors and calculate the average.
def cal_wphi(cond):
    wphi = []
    idx = np.where(cond)[0]
    for i in idx:
        wp = [sv_m[j] * sv_y[j] * kernel(sv_x[i], sv_x[j], type=k_type) \
              for j in range(sv_x.shape[0])]
        wphi.append(np.sum(wp))
    return wphi

b = -(np.max(cal_wphi(sv_y > 0)) + np.min(cal_wphi(sv_y < 0))) / 2.

# Predict the class of test data.
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))
n_test = x_test.shape[0]
n_sv = sv_x.shape[0]
ts_K = np.array([kernel(sv_x[i], x_test[j], type=k_type) 
        for i in range(n_sv) 
        for j in range(n_test)]).reshape(n_sv, n_test)
        
# decision function
y_hat = np.sum(sv_m * sv_y * ts_K, axis=0).reshape(-1, 1) + b
y_pred = np.sign(y_hat)

# Visualize test data and classes.
plt.figure(figsize=(5,5))
test_c = ['red' if a == 1 else 'blue' for a in y_pred]
sv_c = ['red' if a == 1 else 'blue' for a in sv_y]
plt.scatter(x_test[:, 0], x_test[:, 1], s=30, c=test_c, alpha=0.3)
plt.scatter(sv_x[:, 0], sv_x[:, 1], s=100, marker='D', c=sv_c, ec='black', lw=2)
plt.axhline(y=0, lw=1)
plt.axvline(x=0, lw=1)
plt.show()

