# [MXML-4-04] 7.multiclass(softmax_scipy).pyy_prob
# Multiclass classification (Softmax regression)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/D_z48GLwAyM
#
from scipy import optimize
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# Read iris dataset
x, y = load_iris(return_X_y=True)

# one-hot encoding of the y labels.
y_ohe = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()

# Split the data into the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe)

# Add a column vector with all 1 to the feature matrix.
x1_train = np.hstack([np.ones([x_train.shape[0], 1]), x_train])
x1_test = np.hstack([np.ones([x_test.shape[0], 1]), x_test])

REG_CONST = 0.01              # Regularization constant
n_feature = x_train.shape[1]  # The number of features
n_class = y_train.shape[1]    # The number of classes

def softmax(z):
    s = np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)
    return s
    
# Loss function: mean of cross entropy
def ce_loss(W, args):
    train_x = args[0]  # shape=(112,5)
    train_y = args[1]  # shape=(112,3)
    test_x = args[2]
    test_y = args[3]
    W = W.reshape((n_class, n_feature + 1)) # shape=(3, 5)
    
    # Calculate the loss of training data
    z = np.dot(W, train_x.T).T              # shape=(112, 3)
    y_hat = softmax(z)
    train_ce = np.sum(-train_y * np.log(y_hat + 1e-10), axis=1)
    train_loss = train_ce.mean() + REG_CONST * np.mean(np.square(W))

    # Calculate the loss of test data
    # It is independent of training and is measured later to observe changes in loss.
    z = np.dot(W, test_x.T).T
    y_hat = softmax(z)
    test_ce = np.sum(-test_y * np.log(y_hat + 1e-10), axis=1)
    test_loss = test_ce.mean() + REG_CONST * np.mean(np.square(W))
    
    # Save the loss
    trc_train_loss.append(train_loss)
    trc_test_loss.append(test_loss)
        
    return train_loss

# Perform an optimization process
trc_train_loss = []
trc_test_loss = []
init_w = np.ones(n_class * (n_feature + 1)) * 0.1  # shape=(3, 5) → 1D

# constraints: w0 = 0, b0 = 0
def b0_w0(w):
    n = np.arange(n_feature + 1)
    return w[n]

cons = [{'type':'eq', 'fun': b0_w0}]
result = optimize.minimize(ce_loss, init_w,
                           constraints=cons,
                           args=[x1_train, y_train, x1_test, y_test])

# print the result. result.x contains the optimal parameters
print(result)

# Measure the accuracy of test data
W = result.x.reshape(n_class, n_feature + 1)
z = np.dot(W, x1_test.T).T
y_prob = softmax(z)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
acc = (y_pred == y_true).mean()
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

# Check the parameters
w = result.x.reshape((n_class, n_feature + 1))
print('\n', w)

