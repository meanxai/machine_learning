# [MXML-4-04] multiclass(softmax_scipy).py
# Multi-class classification (Softmax regression)
# Use LogisticRegression(multi_class='multinomial')
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/D_z48GLwAyM
#
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Read iris dataset
x, y = load_iris(return_X_y=True)

# Split the data into the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Create a model and fit it to the training data.
# Use multi_class = 'multinomial'
model = LogisticRegression(multi_class='multinomial', max_iter=300)
model.fit(x_train, y_train)

# Predict the classes of the test data
y_pred = model.predict(x_test)

# Measure the accuracy
acc = (y_test == y_pred).mean()
print('\nAccuracy of test data = {:.3f}'.format(acc))

# Check the estimated parameters.
print('\nmodel.coef_ =\n\n', model.coef_)
print('\nmodel.intercept_ =\n\n', model.intercept_)

