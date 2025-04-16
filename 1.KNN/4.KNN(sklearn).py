# [MXML-01-03] 4.KNN(sklearn).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/qZ_6UAVnNMw
#
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Iris dataset.
# x: data, the number of samples=150, the number of features=4
# y: target data with class (0,1,2)
x, y = load_iris(return_X_y=True)

# Split the dataset to training, validation and test data
x_train, x_test, y_train, y_test=train_test_split(x, y, \
                                         test_size = 0.4)
x_val, x_test, y_val, y_test=train_test_split(x_test, y_test,\
                                         test_size = 0.5)
# Z-score normalization
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std # Z-score normalization
x_val = (x_val - mean) / std     # use mean and std from x_train
x_test = (x_test - mean) / std   # use mean and std from x_train

# Set K to sqrt(N)
sqr_k = int(np.sqrt(x_train.shape[0]))

# Build a KNN classification model
knn = KNeighborsClassifier(n_neighbors=sqr_k, metric='minkowski', p=2)

# Model fitting. Since KNN is a lazy learner, no learning is performed 
# at this step. It simply stores the training data points and the 
# parameters.
knn.fit(x_train, y_train)

# Predict the class of validation data.
# The actual learning takes place at this stage, when test or 
# validation data is provided.
y_pred = knn.predict(x_val)

# Measure the accuracy on the validation data
accuracy = (y_val == y_pred).mean()
print('\nK: sqr_K = {}, Accuracy on validation data = {:.3f}'\
      .format(sqr_k, accuracy))

# Determine the optimal K.
# Measure the accuracy on the validation data while changing K.
accuracy = []
for k in range(2, 20):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    accuracy.append((y_val == y_pred).mean())
    
# Find the optimal K value with the highest accuracy.
opt_k = np.array(accuracy).argmax() + 2

# Observe how the accuracy changes as K changes.
plt.plot(np.arange(2, 20), accuracy, marker='o')
plt.xticks(np.arange(2, 20))
plt.axvline(x = opt_k, c='blue', ls = '--')
plt.axvline(x = sqr_k, c='red', ls = '--')
plt.ylim(0.8, 1.1)
plt.title('optimal K = ' + str(opt_k))
plt.show()

# Finally, we use the test data to measure the final performance 
# of the model.
knn = KNeighborsClassifier(n_neighbors = opt_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = (y_test == y_pred).mean()
print('\nK: opt_k = {}, Accuracy on test data = {:.3f}'
      .format(opt_k, accuracy))
   