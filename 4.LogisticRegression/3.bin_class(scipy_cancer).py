# [MXML-4-02] 3.bin_class(scipy_cancer).py
# Breast cancer dataset
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
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Read breast cancer dataset
x, y = load_breast_cancer(return_X_y=True)

# Split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Z-score normalization
# When normalzing the test data, use the mean and standard deviation 
# from the training data.
x_mean = x_train.mean(axis=0).reshape(1, -1)
x_std = x_train.std(axis=0).reshape(1, -1)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# Add a column vector with all 1 to the feature matrix.
# [0.3, 0.4, ...] --> [1.0, 0.3, 0.4, ...]
# [0.1, 0.5, ...] --> [1.0, 0.1, 0.5, ...]
# [ ...]
x1_train = np.hstack([np.ones([x_train.shape[0], 1]), x_train])
x1_test = np.hstack([np.ones([x_test.shape[0], 1]), x_test])

REG_CONST = 0.01  # regularization constant

# Loss function : mean of binary cross entropy
def bce_loss(W, args):
    train_x = args[0]
    train_y = args[1]
    test_x = args[2]
    test_y = args[3]
    
    # Calculate the loss of training data
    y_hat = 1.0 / (1 + np.exp(-np.dot(W, train_x.T)))
    train_bce = -train_y * np.log(y_hat + 1e-10) - (1.0 - train_y) * np.log(1.0 - y_hat + 1e-10)
    train_loss = train_bce.mean() + REG_CONST * np.mean(np.square(W))

    # Calculate the loss of test data
    # It is independent of training and is measured later to observe changes in loss.
    y_hat = 1.0 / (1 + np.exp(-np.dot(W, test_x.T)))
    test_bce = -test_y * np.log(y_hat + 1e-10) - (1.0 - test_y) * np.log(1.0 - y_hat + 1e-10)
    test_loss = test_bce.mean() + REG_CONST * np.mean(np.square(W))
    
    # Save the loss
    trc_train_loss.append(train_loss)
    trc_test_loss.append(test_loss)
        
    return train_loss

# Perform an optimization process
trc_train_loss = []
trc_test_loss = []
init_w = np.ones(x1_train.shape[1]) * 0.1
result = optimize.minimize(fun = bce_loss,
                           x0 = init_w,
                           args=[x1_train, y_train, x1_test, y_test])

# print the result. result.x contains the optimal parameters
print(result)

# Measure the accuracy of test data
y_prob = 1.0 / (1 + np.exp(-np.dot(result.x, x1_test.T)))
y_pred = (y_prob > 0.5).astype('int8')
acc = (y_pred == y_test).mean()
print('\nAccuracy of test data = {:.3f}'.format(acc))

# Visually see that the loss decreases as the iteration progresses
plt.figure(figsize=(5, 4))
plt.plot(trc_train_loss, color='blue', label='train loss')
plt.plot(trc_test_loss, color='red', label='test loss')
plt.legend()
plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

