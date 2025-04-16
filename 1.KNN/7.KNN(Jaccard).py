# [MXML-1-05] 7.KNN(Jaccard).py
# KNN classification on categorical data
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/dDJwm25-_l8
#
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import jaccard_score

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

# x: one-hot encoded or label encoded features
# y: target, k: the number of nearest neighbors
# average: 'binary' or 'macro'
def predict(x, y, k, average):
    match = []
    for t in range(x.shape[0]):
        x_test = x[t]
        y_test = y[t]
        x_train = np.delete(x, t, axis=0)
        y_train = np.delete(y, t, axis=0)
        
        # Compute the Jaccard similarity between a test data point
        # and all training data points.
        similarities = []
        for i in range(x_train.shape[0]):
            J = jaccard_score(x_train[i], x_test, 
                              average=average, zero_division=0.0)
            similarities.append(J)
        
        # Find the k nearest neighbors of the test data point.
        j = np.argsort(similarities)[::-1][:k]
        
        # Predict the class of the test data point by majority vote
        y_pred = np.bincount(y_train[j]).argmax()
        
        # Store whether y_pred and y_test match or not.
        match.append(y_pred == y_test)
        
        print("True class: {}, Predicted class: {}, is match: {}"\
              .format(y_test, y_pred, match[-1]))
    return np.mean(match)  # return the accuracy

# One-hot encoding
ohe = OneHotEncoder().fit_transform(data).toarray().astype('int')
x = ohe[:, :-2]  # one-hot encoded features
y = ohe[:, -1]   # target
K = 5            # 5 nearest neighbors

print("\n* One-hot encoding:")
acc = predict(x, y, K, average='binary')
print("Accuracy: {:.3f}".format(acc))

# Label encoding
le = []
for i in range(data.shape[1]):
    le.append(LabelEncoder().fit_transform(data[:, i]))
le = np.array(le).T

x = le[:, :-1]  # label encoded features
y = le[:, -1]   # target

print("\n* Label encoding:")
acc = predict(x, y, K, average='macro')
print("Accuracy: {:.3f}".format(acc))


