# [MXML-9-03] 4.sklearn(AdaBoost).py
# Test sklearn's AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Read iris dataset
x, y = load_iris(return_X_y=True)

# Create training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Use the decision tree as the base weak learner.
dt = DecisionTreeClassifier(max_depth = 1)

# Generate a AdaBoost model with the SAMME algorithm.
model = AdaBoostClassifier(estimator = dt, 
                           n_estimators = 100,
                           algorithm = 'SAMME')  # default = 'SAMME.R'

# Fit the model to the training data
model.fit(x_train, y_train)

# Predict the class of test data and calculate the accuracy
y_pred = model.predict(x_test)
accuracy = (y_pred == y_test).mean()

print('Accuracy = {:.4f}'.format(accuracy))


