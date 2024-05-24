# [MXML-4-03] 5.multiclass(ovr_1).py
# Multi-class classification (OvR : one vs rest)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/d6FcGZp8AHc
#
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Read iris dataset
x, y = load_iris(return_X_y=True)

# one-hot encoding of the y labels.
y_ohe = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()

# Split the data into the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size = 0.2)

# Perform the OvR. Since there are three labels, three models are used. 
models = []
for m in range(y_train.shape[1]):
    y_sub = y_train[:, m] # y for binary classification
    models.append(LogisticRegression())
    models[-1].fit(x_train, y_sub)

# The labels of the test data are predicted using three trained models.
y_prob = np.zeros(shape=y_test.shape)
for m in range(y_test.shape[1]):
    y_prob[:, m] = models[m].predict_proba(x_test)[:, 1]

# y is predicted as the label with the highest value in y_prob.
y_pred = np.argmax(y_prob, axis=1)

# Measure the accuracy of the test data
y_true = np.argmax(y_test, axis=1)
acc = (y_true == y_pred).mean()
print('Accuracy of test data = {:.3f}'.format(acc))

# Check the estimated parameters.
for m in range(y_test.shape[1]):
    w = models[m].coef_
    b = models[m].intercept_
    print("\nModel-{}:".format(m))
    print("w:", w)
    print("b:", b)
