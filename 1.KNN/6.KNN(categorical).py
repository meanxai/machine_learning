# [MXML-1-05] 6.KNN(categorical).py
#
# This code was used in the machine learning online 
# course provided by www.youtube.com/@meanxai.
#
# A detailed description of this code can be found in
# https://youtu.be/h_wUj1do7qA
#
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# 데이터. columns = [outlook, temp, humidity, windy, play]
d = np.array(
    [['sunny', 'hot', 'high', False, 'no'],
    ['sunny', 'hot', 'high', True, 'no'],
    ['overcast', 'hot', 'high', False, 'yes'],
    ['rainy', 'mild', 'high', False, 'yes'],
    ['rainy', 'cool', 'normal', False, 'yes'],
    ['rainy', 'cool', 'normal', True, 'no'],
    ['overcast', 'cool', 'normal', True, 'yes'],
    ['sunny', 'mild', 'high', False, 'no'],
    ['sunny', 'cool', 'normal', False, 'yes'],
    ['rainy', 'mild', 'normal', False, 'yes'],
    ['sunny', 'mild', 'normal', True, 'yes'],
    ['overcast', 'mild', 'high', True, 'yes'],
    ['overcast', 'hot', 'normal', False, 'yes'],
    ['rainy', 'mild', 'high', True, 'no'],
    ['sunny', 'mild', 'high', True, 'no']])
    
# One-hot encoding
d_ohe = OneHotEncoder().fit_transform(d).toarray().astype('int')
x = d_ohe[:, :-2]  # feature
y = d_ohe[:, -1]   # target

# Create train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y,\
                                              test_size=2)

q = [np.bincount(x_train[:, i], minlength=2) \
     for i in range(x_train.shape[1])]
q_table = np.array(q).T

# Calculate the IOF distance between source (s) and target 
# (t) vector.
# example:
# i =  0  1  2  3  4  5  6  7  8  9
# s = [0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
# t = [1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
#
# q_table:
#     i:   0   1   2   3   4   5   6   7   8   9
# ohe=0: [ 9,  9,  8,  9, 10,  7,  7,  6,  6,  7]
# ohe=1: [ 4,  4,  5,  4,  3,  6,  6,  7,  7,  6]
def iof_distance(s, t):
    sim = np.ones(shape=(s.shape[0],))
    for i in range(s.shape[0]):
        if (s[i] != t[i]):
            log_x = np.log(q_table[s[i], i] + 1e-8)
            log_y = np.log(q_table[t[i], i] + 1e-8)
            sim[i] = (1. / (1. + log_x * log_y))
    return (1. / np.mean(sim)) - 1.

# Calculate the IOF distance between x_test and x_train
# dist = distance matrix, 
# shape = (the number of x_test, the number of x_train)
dist = np.array([iof_distance(s, t) for s in x_test \
                                    for t in x_train])
dist = dist.reshape(x_test.shape[0], -1)

# Select K data with small distance and find the majority of the target 
# class.
K = 5
i_near = dist.argsort(axis=1)[:, :K]
y_near = y_train[i_near]
y_pred = np.array([np.bincount(p).argmax() for p in y_near])

print('true class (y_test):', y_test, end='')
print(',  predict class (y_pred):', y_pred)

