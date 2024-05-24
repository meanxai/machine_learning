# [MXML-8-03] 3.RF_OOB.py
# Add Out-Of-Bag (OOB) score feature to 2.RF(titanic).py.
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/DFh7BefJpfQ
# 
import numpy as np
import pandas as pd
from MyDTreeClassifierRF import MyDTreeClassifierRF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read preprocessed Titanic data.
df = pd.read_csv('data/titanic_clean.csv')

# Survived  Pclass  Sex   Age  SibSp  Parch   Fare  Embarked  Title
#        0       3    1  22.0      1      0   3.62         3      2
#        1       1    0  38.0      1      0  35.64         0      3
#        1       3    0  26.0      0      0   7.92         3      1
#        1       1    0  35.0      1      0  26.55         3      3
#        0       3    1  35.0      0      0   8.05         3      2

y = np.array(df['Survived'])
x = np.array(df.drop('Survived', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y)

N = x_train.shape[0]  # the number of train data points
n_estimators = 50
n_depth = 5        # max_depth of tree
max_features = round(np.sqrt(x_train.shape[1]))  # for column sub sampling

# majority vote for iob_pred, or oob_pred
# p = iob_pred or oob_pred
def majority_vote(p):
    cnt_0 = (p == 0).sum(axis=1)
    cnt_1 = (p == 1).sum(axis=1)
    cnts = np.array([cnt_0, cnt_1])   # shape = (2, 668)
    return np.argmax(cnts, axis=0)

models = []     # base model list
iob_score = []  # Error rate measured with IOB
oob_score = []  # Error rate measured with OOB

# initialize IOB and OOB prediction map
iob_pred = np.ones(shape=(N, n_estimators)) * -1
oob_pred = np.ones(shape=(N, n_estimators)) * -1

# Create n_estimators models
for i in range(n_estimators):
    # Create a Decision Tree for Random Forest
    model = MyDTreeClassifierRF(max_depth=n_depth, 
                                max_samples = N,
                                max_features = max_features)
    
    # train
    p1, p2 = model.fit(x_train, y_train)

    # save trained tree
    models.append(model)
    
    # Create IOB and OOB prediction map
    iob_pred[:, i] = p1
    oob_pred[:, i] = p2
           
    # Calculate IOB and OOB score
    y_trn = majority_vote(iob_pred)
    y_oob = majority_vote(oob_pred)
    
    iob_score.append((y_trn != y_train).mean())
    oob_score.append((y_oob != y_train).mean())

# Visualize IOB and OOB score
plt.figure(figsize=(6, 4))
plt.plot(iob_score, color='blue', lw=1.0, label='IOB error')
plt.plot(oob_score, color='red', lw=1.0, label='OOB error')
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('OOB error rate')
plt.show()

# prediction
y_estimates = np.zeros(shape=(x_test.shape[0], n_estimators))
for i, model in enumerate(models):
    y_estimates[:, i] = model.predict(x_test)

# synthesizing the estimation results
y_prob = y_estimates.mean(axis=1)
y_pred = (y_prob >= 0.5) * 1
accuracy = (y_pred == y_test).mean()
print('\nAccuracy of test data = {:.4f}'.format(accuracy))
print('Final OOB error rate = {:.4f}'.format(oob_score[-1]))

# OOB probability
# In theory, it would be 0.3679.
# This means that x_train is selected with probability 0.6321
# by row subsampling. (1.0 - 0.3679 = 0.6321)
oob_percent = ((oob_pred >= 0).sum(axis=0) / N).mean()
print('OOB probability = {:.4f}'.format(oob_percent))
