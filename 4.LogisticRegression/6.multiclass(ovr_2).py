# [MXML-4-03] 6.multiclass(ovr_2).py
# Multiclass classification (OvR : One-vs-Rest)
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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Read iris dataset
x, y = load_iris(return_X_y=True)

# Split the data into the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Use the multi_class='ovr' function of sklearn's LogisticRegression.
# Even if the 'ovr' is not set, multiclass classification is automatically 
# performed by referring to the number of classes. 
# This was explicitly set to facilitate understanding.
model = LogisticRegression(multi_class='ovr', max_iter=300)
model.fit(x_train, y_train)

# Predict the classes of the test data
y_pred = model.predict(x_test)

# Measure the accuracy of the test data
acc = (y_test == y_pred).mean()
print('\nAccuracy of test data = {:.3f}'.format(acc))

# Check the estimated parameters.
print('\nmodel.coef_ =\n\n', model.coef_)
print('\nmodel.intercept_ =\n\n', model.intercept_)

