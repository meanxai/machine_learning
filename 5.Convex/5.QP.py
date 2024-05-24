# [MXML-5-04] 5.QP.py
# QP problem with an equality and an inequality constraints.
# https://cvxopt.org/examples/tutorial/qp.html
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/_5QuyiCI1rc
#
# min. 2 * x1^2 + x2^2 + x1 * x2 + x1 + x2
# s.t. x1 >= 0
#      x2 >= 0
#      x1 + x2 = 1
#
# QP standard form
# minimize   1/2 * xT.P.x + qT.x
# subject to G.x <= h
#            A.x = b
# 
# min. 1/2 [x1 x2][4 1][x1] + [1 1][x1]
#                 [1 2][x2]        [x2]
#
# s.t. [-1  0][x1] <= [0]
#      [ 0 -1][x2]    [0]
#      
#      [1 1][x1] = 1
#           [x2]
#
# x = [x1] P = [4 1] q = [1] G = [-1  0] h = [0] A = [1 1] b = 1
#     [x2]     [1 2]     [1]     [ 0 -1]     [0]
from cvxopt import matrix, solvers
import numpy as np

P = matrix(np.array([[4, 1], [1, 2]]), tc='d')
q = matrix(np.array([[1], [1]]), tc='d')
G = matrix(np.array([[-1, 0],[0, -1]]), tc='d')
h = matrix(np.array([[0], [0]]), tc='d')
A = matrix(np.array([[1, 1]]), tc='d')
b = matrix(1, tc='d')

sol = solvers.qp(P, q, G, h, A, b)

p_star = sol['primal objective']
x1, x2 = sol['x']
y = sol['y'][0]     # Lagrange multiplier for x1 + x2 = 1
z1 = sol['z'][0]    # Lagrange multiplier for -x1 <= 0
z2 = sol['z'][1]    # Lagrange multiplier for -x2 <= 0
gap = sol['gap']    # duality gap

# z and y are Lagrange multipliers.
# L = (1/2) * xT.P.x + qT.x + zT(G.x - h) + yT(A.x - b)
# zT = z-transpose, yT = y-transpose
print('\nx1 = {:.3f}'.format(x1))
print('x2 = {:.3f}'.format(x2))
print('y = {:.3f}'.format(y))
print('z1 = {:.3f}'.format(z1))
print('z2 = {:.3f}'.format(z2))
print('p* = {:.3f}'.format(p_star))
print('duality gap = {:.3f}'.format(gap))