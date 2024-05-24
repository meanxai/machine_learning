# [MXML-2-07] 2.CART(classification).py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/gct9gGOvPek
#
import numpy as np
import pandas as pd
from MyDTreeClassifier import MyDTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pprint

# Read the Titanic dataset and perform simple preprocessing.
df = pd.read_csv('data/titanic.csv')
df['Age'].fillna(df['Age'].mean(), inplace = True)  # Replace with average
df['Embarked'].fillna('N', inplace = True)          # Replace with 'N'
df['Sex'] = df['Sex'].factorize()[0]                # label encoding
df['Embarked'] = df['Embarked'].factorize()[0]      # label encoding
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#  Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked
# 0       0       3    0  22.0      1      0   7.2500         0
# 1       1       1    1  38.0      1      0  71.2833         1
# 2       1       3    1  26.0      0      0   7.9250         0
# 3       1       1    1  35.0      1      0  53.1000         0
# 4       0       3    0  35.0      0      0   8.0500         0

# split the data into train, validation and test data.
y = np.array(df['Survived'])
x = np.array(df.drop('Survived', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y)

depth = 3
my_model = MyDTreeClassifier(max_depth = depth)
my_model.fit(x_train, y_train)
my_pred = my_model.predict(x_test)
acc = (y_test == my_pred).mean()
print('MyTreeClassifier: accuracy = {:.3f}'.format(acc))

# Compare the results with sklearn's DecisionTreeClassifier.
# ----------------------------------------------------------
sk_model = DecisionTreeClassifier(max_depth=depth, 
                                  random_state=1)
sk_model.fit(x_train, y_train)
sk_pred = sk_model.predict(x_test)
acc = (y_test == sk_pred).mean()
print('DecisionTreeClassifier: accuracy = {:.3f}'.format(acc))

print('\nMyTreeClassifier: estimator2:')
pprint.pprint(my_model.estimator2, sort_dicts=False)

plt.figure(figsize=(12, 6))
tree.plot_tree(sk_model)
plt.show()
