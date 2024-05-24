# [MXML-8-02]: 2.RF(sklearn).py
# Implement Random Forest using scikit-learn.
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/9si5fELmtg0
# 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Read preprocessed Titanic data.
df = pd.read_csv('data/titanic_clean.csv')
y = np.array(df['Survived'])
x = np.array(df.drop('Survived', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y)

n_estimators = 100
n_depth = 3                           # max_depth of tree

# Implement Random Forest using DecisionTreeClassifier
models = []  # base model list
n = x_train.shape[0]  # the number of train data points
for i in range(n_estimators):
    # row subsampling
    i_row = np.random.choice(np.arange(0, n), n, replace=True)
    x_sample = x_train[i_row, :]
    y_sample = y_train[i_row]

    # Create a tree for Random Forest
    # Column subsampling for each split is performed within the model.
    model = DecisionTreeClassifier(max_depth=n_depth,
                                   max_features="sqrt")
    
    # train the tree
    model.fit(x_sample, y_sample)
    
    # save trained tree
    models.append(model)

# prediction
y_estimates = np.zeros(shape=(x_test.shape[0], n_estimators))
for i, model in enumerate(models):
    y_estimates[:, i] = model.predict(x_test)

# synthesizing the estimation results
y_prob = y_estimates.mean(axis=1)
y_pred = (y_prob >= 0.5) * 1
print('\nAccuracy1 = {:.4f}'.format((y_pred == y_test).mean()))

# Implement Random Forest using RandomForestClassifier
model = RandomForestClassifier(n_estimators=n_estimators,
                               max_depth=n_depth,
                               max_samples=n,       # default
                               max_features="sqrt") # default
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nAccuracy2 = {:.4f}'.format((y_pred == y_test).mean()))

model.estimators_
# [DecisionTreeClassifier(max_depth=3, max_features='sqrt',
#                         random_state=1090277217),
#  DecisionTreeClassifier(max_depth=3, max_features='sqrt',
#                         random_state=1758239483),
#  DecisionTreeClassifier(max_depth=3, max_features='sqrt',
#                         random_state=1420256802)
# ...