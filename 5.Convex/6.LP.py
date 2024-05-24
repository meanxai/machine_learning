# [MXML-5-04] 6.LP.py
# https://cvxopt.org/examples/tutorial/lp.html
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/_5QuyiCI1rc
# 
# min. 2 * x1 + x2
# s.t. -x1 + x2 <= 1
#       x1 + x2 >= 2    --> -x1 - x2 <= -2
#            x2 >= 0    --> -x2 <= 0
#       x1 - 2 * x2 <= 4
#       x1 - 5 * x2 = 15
#
# LP standard form
# minimize   cT.x
# subject to G.x <= h
#            A.x = b
# 
# min. [2 1][x1]
#           [x2]
#
# s.t. G.x <= h   [-1  1][x1] <= [ 1]
#                 [-1 -1][x2]    [-2]
#                 [ 0 -1]        [ 0]
#                 [ 1 -2]        [ 4]
#      
#      A.x = b    [1 -5][x1] = 15
#                       [x2]
#
# x = [x1] c = [2] G = [-1  1] h = [ 1] A = [1 1] b = 1
#     [x2]     [1]     [-1 -1]     [-2]
#                      [ 0 -1]     [ 0]
#                      [ 1 -2]     [ 4]
from cvxopt import matrix, solvers
import numpy as np

c = matrix(np.array([[2], [1]]), tc='d')
G = matrix(np.array([[-1, 1],[-1, -1],[0, -1],[1, -2]]), tc='d')
h = matrix(np.array([[1], [-2], [0], [4]]), tc='d')
A = matrix(np.array([[1, -5]]), tc='d')
b = matrix(1, tc='d')
sol = solvers.lp(c, G, h, A, b)

p_star = sol['primal objective']
x1, x2 = sol['x']
y = sol['y'][0]     # Lagrange multiplier for A.x = b
z1 = sol['z'][0]    # Lagrange multiplier for G1.x <= h1
z2 = sol['z'][1]    # Lagrange multiplier for G2.x <= h2
z3 = sol['z'][2]    # Lagrange multiplier for G3.x <= h3
z4 = sol['z'][3]    # Lagrange multiplier for G4.x <= h4
gap = sol['gap']    # duality gap

# z and y are Lagrange multipliers.
# L = cT.x + zT(G.x - h) + yT(A.x - b)
# zT = z-transpose, yT = y-transpose
print('\nx1 = {:.3f}'.format(x1))
print('x2 = {:.3f}'.format(x2))
print('y = {:.3f}'.format(y))
print('z1 = {:.3f}'.format(z1))
print('z2 = {:.3f}'.format(z2))
print('z3 = {:.3f}'.format(z3))
print('z4 = {:.3f}'.format(z4))
print('p* = {:.3f}'.format(p_star))
print('duality gap = {:.3f}'.format(gap))
