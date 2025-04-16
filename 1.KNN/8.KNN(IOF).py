# [MXML-1-06] 8.KNN(IOF).py
# KNN classification for categorical data using IOF distance
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/D3680EliTzA
#
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Golf play dataset
# data source: 
# https://www.kaggle.com/datasets/priy998/golf-play-dataset
# columns = [outlook, temperature, humidity, windy, play]
data = np.array(
        [['sunny',    'hot',  'high',   False, 'no'],
         ['sunny',    'hot',  'high',   True,  'no'],
         ['overcast', 'hot',  'high',   False, 'yes'],
         ['rainy',    'mild', 'high',   False, 'yes'],
         ['rainy',    'cool', 'normal', False, 'yes'],
         ['rainy',    'cool', 'normal', True,  'no'],
         ['overcast', 'cool', 'normal', True,  'yes'],
         ['sunny',    'mild', 'high',   False, 'no'],
         ['sunny',    'cool', 'normal', False, 'yes'],
         ['rainy',    'mild', 'normal', False, 'yes'],
         ['sunny',    'mild', 'normal', True,  'yes'],
         ['overcast', 'mild', 'high',   True,  'yes'],
         ['overcast', 'hot',  'normal', False, 'yes'],
         ['rainy',    'mild', 'high',   True,  'no'],
         ['sunny',    'mild', 'high',   True,  'no']])
    
# Compute the IOF similarity between a test data point and  
# a training data point
# example:
#     i =  0  1  2  3  4  5  6  7  8  9
# train = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
# test  = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
#
# q_table:
#     i:   0   1   2   3   4   5   6   7   8   9
# ohe=0: [10,  9,  9, 10, 11,  7,  7,  7,  7,  7]
# ohe=1: [ 4,  5,  5,  4,  3,  7,  7,  7,  7,  7]
def iof_similarity(train, test, q_table):
    sim = np.ones(shape=(train.shape[0],))
    for i in range(train.shape[0]):
        if (train[i] != test[i]):
            log_x = np.log(q_table[train[i], i] + 1)
            log_y = np.log(q_table[test[i], i] + 1)
            sim[i] = (1. / (1. + log_x * log_y))
    return np.mean(sim)

# x: one-hot encoded or label encoded features
# y: target, k: the number of nearest neighbors
def predict(x, y, k):
    match = []
    for t in range(x.shape[0]):
        x_test = x[t]
        y_test = y[t]
        x_train = np.delete(x, t, axis=0)
        y_train = np.delete(y, t, axis=0)
        n = np.unique(x_train).shape[0]
        
        # An occurrence frequency table
        q = [np.bincount(x_train[:, i], minlength=n) \
             for i in range(x_train.shape[1])]
        q_table = np.array(q).T
        
        # IOF similarity
        similarities = [iof_similarity(train, x_test, q_table)\
                        for train in x_train]

        # Find the k nearest neighbors of the test data point.
        j = np.argsort(similarities)[::-1][:k]
        
        # Predict the class of the x_test by majority vote
        y_pred = np.bincount(y_train[j]).argmax()
        
        # Store whether y_pred and y_test match or not.
        match.append(y_pred == y_test)
        
        print("True class: {}, Predicted class: {}, match: {}"\
              .format(y_test, y_pred, match[-1]))
    return np.mean(match)  # return the accuracy

# One-hot encoding
ohe = OneHotEncoder().fit_transform(data).toarray().astype('int')
x = ohe[:, :-2]  # one-hot encoded features
y = ohe[:, -1]   # target
K = 3            # 3 nearest neighbors

print("\n* One-hot encoding:")
acc = predict(x, y, K)
print("\nAccuracy: {:.3f}".format(acc))

# Label encoding
le = []
for i in range(data.shape[1]):
    le.append(LabelEncoder().fit_transform(data[:, i]))
le = np.array(le).T

x = le[:, :-1]  # label encoded features
y = le[:, -1]   # target

print("\n* Label encoding:")
acc = predict(x, y, K)
print("\nAccuracy: {:.3f}".format(acc))

