# [MXML-2-10]: 5.CART(multiclass).py
# Multiclass classification test code
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/o43mZv_Cmxw
#
import numpy as np
from sklearn.datasets import load_iris
from MyDTreeClassifier import MyDTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load iris dataset
# x: data, the number of samples=150, the number of features=4
# y: target data with class (0,1,2)
x, y = load_iris(return_X_y=True)

# Generate training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Model-1: using our model - refer to [MXML-2-07] video
model1 = MyDTreeClassifier(max_depth=3)
model1.fit(x_train, y_train)

# Estimate the class of validation date.
y_pred1 = model1.predict(x_test)

# Measure the accuracy for validation data
accuracy1 = (y_test == y_pred1).mean()
print('\nAccuracy of Model-1 = {:.3f}'.format(accuracy1))

# Model-2: using sklearn
model2 = DecisionTreeClassifier(max_depth=3)
model2.fit(x_train, y_train)

# Estimate the class of validation date.
y_pred2 = model2.predict(x_test)

# Measure the accuracy for validation data
accuracy2 = (y_test == y_pred2).mean()
print('Accuracy of Model-2 = {:.3f}'.format(accuracy2))

print("\nModel-1: y_pred1")
print(y_pred1)
print("\nModel-2: y_pred2")
print(y_pred2)

