# [MXML-2-03] 1.ID3(titanic_part).py
# ID3/C4.5 decision tree test code
# CART is widely used than ID3/C4.5. Sklearn  supports CART.
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/m3o0-K07gLI
#
#
# I used the package below to test ID3/C4.5.
# https://github.com/svaante/decision-tree-id3
# pip install decision-tree-id3
# pip install pydot
# pip install graphviz
# sudo apt install graphviz
# -----------------------------------------------------------

# "from sklearn.externals import six" is used for id3, but "six"
# is missing in the sklearn.externals, resulting in the following 
# error: cannot import name "six" from 'sklearn.externals'
# Add following to prevent errors.
import six
import sys; sys.modules['sklearn.externals.six'] = six
import pandas as pd
from id3 import Id3Estimator
from id3 import export_graphviz
import pydot
from sklearn.model_selection import train_test_split

# Use just 3 features in the Titanic dataset:
feat_names = ['Pclass', 'Sex', 'Age']
df = pd.read_csv('data/titanic.csv')[feat_names + ['Survived']]
df = df.dropna().reset_index()
df.info()

# Separate the data into feature and target.
x_data = df[feat_names].copy()
y_data = df['Survived']

# Convert string (Sex) to number. female = 0, male = 1
x_data['Sex'] = x_data['Sex'].map({'female':0, 'male':1})

# Convert real numbers (Age) to 4 categories.
x_data['Age'] = pd.qcut(x_data['Age'], 4, labels=False)

# Split the data into training and test data.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

# Build ID3/C4.5 decision tree.
estimator = Id3Estimator(gain_ratio=True, prune=True)
estimator = estimator.fit(x_train, y_train, check_input=False)

# Evaluate performance with test data.
y_pred = estimator.predict(x_test)
acc = (y_pred == y_test).mean()
print('\nAccuracy of test data = {:.4f}'.format(acc))

# Evaluate performance with training data.
y_pred = estimator.predict(x_train)
acc = (y_pred == y_train).mean()
print('Accuracy of train data = {:.4f}\n'.format(acc))

# Visualize the tree result
tree = export_graphviz(estimator.tree_, 'id3_tree.dot', feat_names)
(graph,) = pydot.graph_from_dot_file('id3_tree.dot')
graph.write_png('id3_tree.png')
!nomacs 'id3_tree.png'   # Check the tree image with the image viewer.
