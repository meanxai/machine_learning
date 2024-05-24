# [MXML-5-03] 3.IQP_1.py
# Inequality constrained QP (IQP-1)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/yn04TeRxKko
#
# 
# Least squares problem:
# minimize   x1^2 + x2^2
# subject to x1 + x2 <= 1
#
# QP standard form
# minimize   1/2 * xT.P.x + qT.x
# subject to G.x <= h
#            A.x = b
# 
# min. 1/2 [x1 x2][2 0][x1] + [0 0][x1]
#                 [0 2][x2]        [x2]
#
# s.t. [1 1][x1] <= 1
#           [x2]
#
# x = [x1]  P = [2 0]  q = [0]  G = [1 1]  h = 1
#     [x2]      [0 2]      [0]
from cvxopt import matrix, solvers
import numpy as np

P = matrix(np.array([[2, 0], [0, 2]]), tc='d')
q = matrix(np.array([[0], [0]]), tc='d')
G = matrix(np.array([[1, 1]]), tc='d')
h = matrix(1, tc='d')

sol = solvers.qp(P, q, G, h)

p_star = sol['primal objective']
x1, x2 = sol['x']
z = sol['z'][0]     # Lagrange multiplier for G.x <= h
gap = sol['gap']    # duality gap
s = sol['s'][0]     # slack variable

# z and y are Lagrange multipliers. y is not used here.
# L = (1/2) * xT.P.x + qT.x + zT(G.x - h) + yT(A.x - b)
# zT = z-transpose, yT = y-transpose
print('\nx1 = {:.3f}'.format(x1))
print('x2 = {:.3f}'.format(x2))
print('z = {:.3f}'.format(z))
print('s = {:.3f}'.format(s))
print('p* = {:.3f}'.format(p_star))
print('duality gap = {:.3f}'.format(gap))

