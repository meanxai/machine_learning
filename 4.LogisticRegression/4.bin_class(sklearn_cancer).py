# [MXML-4-02] 4.bin_class(sklearn_cancer).py
# Using sklearn's LogisticRegression()
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/MifHxwJYOyU
#
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Read breast cancer dataset
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

# Split the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Z-score normalization
# When normalzing the test data, use the mean and standard deviation 
# from the training data.
x_mean = x_train.mean(axis=0).reshape(1, -1)
x_std = x_train.std(axis=0).reshape(1, -1)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# regularization constant (strenth)
REG_CONST = 0.01

# Create a model and fit it to the training data.
# C: inverse of regularization strength
model = LogisticRegression(penalty='l2', C=1./REG_CONST, max_iter=300)
model.fit(x_train, y_train)

# Predict the classes of test data and measure the accuracy of test data
y_pred = model.predict(x_test)
acc = (y_pred == y_test).mean()
print('\nAccuracy of test data = {:.3f}'.format(acc))
