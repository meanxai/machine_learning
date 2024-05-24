# [MXML-6-06] 7.SVC(kernel_trick).py
# Implement nonlinear SVM using SVC.
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
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 4 data samples. 2 ‘+’ samples, 2 ‘-’ samples
x = np.array([[0., 1.], [1., 1.], [1., 0.], [0., 0.]])
y = np.array([-1., 1., -1., 1.])

C = 1.0
# model = SVC(C=C, kernel='rbf', gamma=0.5)
model = SVC(C=C, kernel='poly', degree=3)
model.fit(x, y)

# Intercept (b)
# w = model.coef_[0]
# AttributeError: coef_ is only available when using a linear kernel
b = model.intercept_[0]

# Predict the class of test data.
x_test = np.random.uniform(-0.5, 1.5, (1000, 2))

# decision function
y_hat = model.decision_function(x_test)
y_pred = np.sign(y_hat)
# y_pred = model.predict(x_test)  # It is the same as above.

# Visualize test data and classes.
plt.figure(figsize=(5,5))
test_c = ['red' if a == 1 else 'blue' for a in y_pred]
plt.scatter(x_test[:, 0], x_test[:, 1], s=30, c=test_c, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], s=100, marker='D', c='white', ec='black', lw=2)
plt.axhline(y=0, lw=1)
plt.axvline(x=0, lw=1)
plt.show()

