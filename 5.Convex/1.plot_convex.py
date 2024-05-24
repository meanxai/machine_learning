# [MXML-5-01] 1.plot_convex.py (Plot 3D convex function)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/8BiHfVrdClU
#
import matplotlib.pyplot as plt
import numpy as np

# f(x)
def f_xy(x1, x2):
    return (x1 ** 2) + (x2 ** 2)
    # return 3 * x1 + x2
    # return (x1 ** 2) + x2 * (x1 - 1)
    # return 2 * (x1 ** 2) + (x2 ** 2) + x1 * x2 + x1 + x2
    # return -5 * x1 / 3 - x2 + 5

t = 0.1
x, y = np.meshgrid(np.arange(-10, 10, t), np.arange(-10, 10, t))
zs = np.array([f_xy(a, b) for [a, b] in zip(np.ravel(x), np.ravel(y))])
z = zs.reshape(x.shape)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# surface를 그린다.
ax.plot_surface(x, y, z, alpha=0.7)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.azim = -50
ax.elev = 30
plt.show()


