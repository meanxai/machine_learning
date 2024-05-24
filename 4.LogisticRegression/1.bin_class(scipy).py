# [MXML-4-02] 1.bin_class(scipy).pyt
# Logistic Regression : binary classification
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/MifHxwJYOyU
#
from scipy import optimize
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a simple dataset for binary classification
def bin_class_data(n):
    n1 = int(n / 2)
    a = np.random.normal(-1.0, 1.0, n1)
    b = np.random.normal(1.0, 1.0, n1)
    x = np.hstack([a, b]).reshape(-1, 1)
    y = np.hstack([np.zeros(n1), np.ones(n1)])
    return x, y
    
x, y = bin_class_data(n=1000)  # create 1000 data points
X = np.hstack([np.ones([x.shape[0], 1]), x])
y = y.astype('int8')

# Visually check the data
plt.scatter(x, y, c='r', s=10, alpha=0.5)
plt.show()

# Split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Loss function : mean of binary cross entropy
def bce_loss(W, args):
    tx = args[0]
    ty = args[1]
    trc = args[2]
    y_hat = 1.0 / (1 + np.exp(-np.dot(W, tx.T)))
    bce = -ty * np.log(y_hat + 1e-8) - (1.0 - ty) * np.log(1.0 - y_hat + 1e-8)
    loss = bce.mean()

    # save the loss
    if trc == True:
        trace_W.append([W, loss])
    return loss

# Perform an optimization process
trace_W = []
result = optimize.minimize(fun = bce_loss,
                            x0 = [-5, 15],
                            args=[x_train, y_train, True])

# print the result. result.x contains the optimal parameters
print(result)

# Visually check the data and the predicted regression curves
y_hat = 1.0 / (1 + np.exp(-np.dot(result.x, x_train.T)))
plt.figure(figsize=(5, 4))
plt.scatter(x, y, s=5, c='r', label = 'data')
plt.scatter(x_train[:, 1], y_hat, c='blue', s=1, label = 'sigmoid')
plt.legend()
plt.axhline(y = 0.5, linestyle='--', linewidth=0.5)
plt.show()

# Measure the accuracy of test data
y_prob = 1.0 / (1 + np.exp(-np.dot(result.x, x_test.T)))
y_pred = (y_prob > 0.5).astype('int8')
acc = (y_pred == y_test).mean()
print('\nAccuracy of test data = {:.3f}'.format(acc))

# Visually check the loss function and the path to the optimal point
w0, w1 = np.meshgrid(np.arange(-20, 20, 1), np.arange(-5, 20, 1))
zs = np.array([bce_loss(np.array([a, b]), [x_train, y_train, False]) \
               for [a, b] in zip(np.ravel(w0), np.ravel(w1))])
z = zs.reshape(w0.shape)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Drawing the surface of the loss function
ax.plot_surface(w0, w1, z, alpha=0.7)

# Drawing the path to the optimal point
b = np.array([tw0 for [tw0, tw1], td in trace_W])
w = np.array([tw1 for [tw0, tw1], td in trace_W])
d = np.array([td for [tw0, tw1], td in trace_W])
ax.plot(b[0], w[0], d[0], marker='x', markersize=15, color="r")
ax.plot(b[-1], w[-1], d[-1], marker='*', markersize=20, color="r")
ax.plot(b, w, d, marker='o', color="r")

ax.set_xlabel('W0 (bias)')
ax.set_ylabel('W1 (slope)')
ax.set_zlabel('cross entropy')
ax.azim = 50
ax.elev = 50    # [50, 0]
plt.show()

# Visually see that the loss decreases as the iteration progresses
plt.figure(figsize=(5, 4))
plt.plot([e for w, e in trace_W], color='red')
plt.title('train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
