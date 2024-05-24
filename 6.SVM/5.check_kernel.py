# [MXML-6-05] 5.check_kernel.py
# For arbitrary real data, if the eigenvalues ​​of the kernel
# matrix (K) are all non-negative, then K is positive semi-definite
# (PSD) and is a valid kernel function.
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/NiuJihA05Ds
# 
import numpy as np

# random dataset (2-dims)
x = np.random.rand(100, 2)
n = x.shape[0]

# kernel functions
rbf_kernel = lambda a, b: np.exp(-np.linalg.norm(a - b)**2 / 2)
pol_kernel = lambda a, b: (1 + np.dot(a, b)) ** 2
sig_kernel = lambda a, b: np.tanh(3 * np.dot(a, b) + 5)
cos_kernel = lambda a, b: np.cos(np.dot(a, b))
kernels = [rbf_kernel, pol_kernel, sig_kernel, cos_kernel]
names = ['RBF', 'Polynomial', 'Sigmoid', 'Cos']

for kernel, name in zip(kernels, names):
    # Kernel matrix (Gram matrix).
    K = np.array([kernel(x[i], x[j]) 
                  for i in range(n) 
                  for j in range(n)]).reshape(n, n)
    
    # Find eigenvalues, eigenvectors
    w, v = np.linalg.eig(K)

    # The function defined above is a valid kernel if all
    # eigenvalues ​​of K are non-negative.
    print('\nKernel : ' + name)
    print('max eigenvalue =', w.max().round(3))
    print('min eigenvalue =', w.min().round(8))
    
    if w.min().real > -1e-8:
        print('==> valid kernel')
    else:
        print('==> invalid kernel')

