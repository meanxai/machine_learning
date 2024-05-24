# [MXML-8-05] 5.RF_proximity.py
# Add missing value estimation functionality to RandomForestClassifier.
#
# To understand this code, you need to learn about the proximity matrix 
# from the previous video, [MXML-8-04].
#
# reference : 
# [1] Random Forests, Leo Breiman and Adele Cutler
# [2] https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
# [3] https://www.youtube.com/watch?v=sQ870aTKqiM
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/tuq2stgBktQ
# 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read Titanic dataset
df = pd.read_csv('data/titanic.csv')
df['Embarked'].fillna('N', inplace = True)          # Replace with 'N'
df['Sex'] = df['Sex'].factorize()[0]                # encoding
df['Embarked'] = df['Embarked'].factorize()[0]      # encoding
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df.info()

# RangeIndex: 891 entries, 0 to 890
# Data columns (total 8 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   Survived  891 non-null    int64  
#  1   Pclass    891 non-null    int64  
#  2   Sex       891 non-null    int64  
#  3   Age       714 non-null    float64
#  4   SibSp     891 non-null    int64  
#  5   Parch     891 non-null    int64  
#  6   Fare      891 non-null    float64
#  7   Embarked  891 non-null    int64  
 
#    Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked
# 0         0       3    0  22.0      1      0   7.2500         0
# 3         1       1    1  35.0      1      0  53.1000         0
# 4         0       3    0  35.0      0      0   8.0500         0
# 5         0       3    0   NaN      0      0   8.4583         2
# 6         0       1    0  54.0      0      0  51.8625         0
# 8         1       3    1  27.0      0      2  11.1333         0

# create training and test data
y = np.array(df['Survived'])
x = np.array(df.drop('Survived', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initially, missing values ​​in 'Age' are replaced with the average value.
AGE = 2  # column number of 'Age' feature

# the position of missing values. It is for later use.
i_train = np.where(np.isnan(x_train[:, AGE]))[0]  # training data
i_test = np.where(np.isnan(x_test[:, AGE]))[0]    # test data

# indices where y_train=0 and y_train=1
i_y0 = np.where(y_train == 0)[0]
i_y1 = np.where(y_train == 1)[0]

# the mean value of 'Age' where y_train=0, and the same where y_train=1.
y0_mean = np.nanmean(x_train[i_y0, AGE]) # where y_train=0
y1_mean = np.nanmean(x_train[i_y1, AGE]) # where y_train=1

# replace nan in 'Age' where y_train = 0 to y0_mean
x_train[i_y0, AGE] = np.nan_to_num(x_train[i_y0, AGE], nan=y0_mean)

# replace nan in 'Age' where y_train = 1 to y1_mean
x_train[i_y1, AGE] = np.nan_to_num(x_train[i_y1, AGE], nan=y1_mean)

# print('Before:\n', x_train.round(2))
# print('y0_mean = {:.2f}, y1_mean = {:.2f}'.format(y0_mean, y1_mean))
plt.plot(x_train[i_train, AGE], 'bo')
plt.title('Initial values for the missing values')
plt.show()

print('Before:\n', x_train.round(2))
print('y0_mean = {:.2f}, y1_mean = {:.2f}'.format(y0_mean, y1_mean))


# Create Proximity matrix
# normalize = 0: pm / n_tree
# normalize ≠ 0: Normalize columns to sum to 1
def proximity_matrix(model, x, normalize=0):
    n_tree = len(model.estimators_)
    
    # Apply trees in the forest to X, return leaf indices.
    leaf = model.apply(x)  # shape = (x.shape[0], n_tree)
    
    pm = np.zeros(shape=(x.shape[0], x.shape[0]))
    for i in range(n_tree):
        t = leaf[:, i]
        pm += np.equal.outer(t, t) * 1.

    np.fill_diagonal(pm, 0)    
    if normalize == 0:
        return pm / n_tree
    else:
        return pm / pm.sum(axis=0, keepdims=True)

n_estimators = 50
n_depth = 5

# Missing value imputation using the proximity matrix
for i in range(5):   # 5 iterations
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=n_depth,
                                   oob_score=True)
    model.fit(x_train, y_train)
    
    # Create proximity matrix
    pm = proximity_matrix(model, x_train, normalize=1)
    
    # estimate the missing values of 'Age' using the proximity matrix
    x_age = x_train[:, AGE].copy()
    u_age = np.dot(x_age, pm)       # updated 'Age'
    x_train[i_train, AGE] = u_age[i_train]

print('\nAfter:\n', x_train.round(2))
plt.plot(x_train[i_train, AGE], 'ro')
plt.title('Estimated values for the missing values')
plt.show()

# Train a new model after imputing missing values of training data
model = RandomForestClassifier(n_estimators=n_estimators,
                               max_depth=n_depth,
                               oob_score=True)
model.fit(x_train, y_train)

# Predict the test data. There are also missing values ​​in 'Age' in the test data.
#
# [2] Proximities
# When a test set is present, the proximities of each case in the test set 
# with each case in the training set can also be computed.

# Initially, there is no target value y in the test data, so the missing values ​​
# are replaced with the mean value of the training data.
x_test[i_test, AGE] = x_train[:, AGE].mean()

x_data = np.vstack([x_train, x_test]) # combine training and test data
pm = proximity_matrix(model, x_data, normalize=1)
x_age = x_data[:, AGE].copy()         # 'Age' feature data
u_age = np.dot(x_age, pm)             # updated 'Age' feature

u_age = u_age[-x_test.shape[0]:]      # 'Age' of test data
u_test = x_data[-x_test.shape[0]:]    # test data
u_test[i_test, AGE] = u_age[i_test]   # update the missing values in test data

# predict
y_pred = model.predict(u_test)

print('\nResults:')
print('Accuracy = {:.4f}'.format((y_pred == y_test).mean()))
print('Final OOB error rate = {:.4f}'.format(1 - model.oob_score_))
