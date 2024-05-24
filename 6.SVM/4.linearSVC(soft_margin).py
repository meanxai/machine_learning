# [MXML-6-04] 4.linearSVC(soft_margin).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/LdOcJfJTcwU
# 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# training data
x = np.array([[0.2,   0.869],
              [0.687, 0.212],
              [0.822, 0.411],
              [0.738, 0.694],
              [0.176, 0.458],
              [0.306, 0.753],
              [0.936, 0.413],
              [0.215, 0.410],
              [0.612, 0.375],
              [0.784, 0.602],
              [0.612, 0.554],
              [0.357, 0.254],
              [0.204, 0.775],
              [0.512, 0.745],
              [0.498, 0.287],
              [0.251, 0.557],
              [0.502, 0.523],
              [0.119, 0.687],
              [0.495, 0.924],
              [0.612, 0.851]])

y = np.array([-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1])

C = 50
model = LinearSVC(penalty='l2', loss='hinge', C=C)
# model = LinearSVC(penalty='l2', loss='squared_hinge', C=C)
model.fit(x, y)

# parameters
w = model.coef_[0]
b = model.intercept_[0]

# Visualize the data points
plt.figure(figsize=(7,7))
color= ['red' if a == 1 else 'blue' for a in y]
plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
plt.xlim(0, 1)
plt.ylim(0, 1)

# Visualize the decision boundary
x1_dec = np.linspace(0, 1, 50).reshape(-1, 1)
x2_dec = -(w[0] / w[1]) * x1_dec - b / w[1]
plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')

# Visualize the positive & negative boundary
w_norm = np.sqrt(np.sum(w ** 2))
w_unit = w / w_norm
half_margin = 1 / w_norm
upper = np.hstack([x1_dec, x2_dec]) + half_margin * w_unit
lower = np.hstack([x1_dec, x2_dec]) - half_margin * w_unit

plt.plot(upper[:, 0], upper[:, 1], '--', lw=1.0, label='positive boundary')
plt.plot(lower[:, 0], lower[:, 1], '--', lw=1.0, label='negative boundary')

# display slack variables, slack variable = max(0, 1 - y(wx + b))
y_hat = np.dot(w, x.T) + b
slack = np.maximum(0, 1 - y * y_hat)
for s, (x1, x2) in zip(slack, x):
    plt.annotate(str(s.round(2)), (x1-0.02, x2 + 0.03))

# Visualize support vectors.
sv = x[np.where(np.abs(y_hat) <= 1.0)[0]]
plt.scatter(sv[:, 0], sv[:, 1], s=30, c='white')
    
plt.title('C = ' + str(C) + ',  Σξ = ' + str(np.sum(slack).round(2)))
plt.legend()
plt.show()

# Hinge & squared hinge loss plot for [+] samples (y = +1)
x_rand = np.random.rand(100, 2)
y_rand = np.dot(w, x_rand.T) + b     # y_hat for x_rand
s_rand = np.maximum(0, 1 - y_rand)   # slack variables for y_rand

sort_idx = np.argsort(y_rand)
y_rand = y_rand[sort_idx]
s_rand = s_rand[sort_idx]

plt.plot(y_rand, s_rand, c='blue', label='Hinge loss')
plt.plot(y_rand, s_rand ** 2, c='red', label='Squared hinge loss')
plt.legend()
plt.axvline(x=0, lw=1)
plt.axvline(x=1, lw=1)
plt.xlabel('y_hat')
plt.ylabel('ξ')
plt.ylim(0, 4)
plt.title('Hinge & squared hinge loss for (+) sample')
plt.show()
