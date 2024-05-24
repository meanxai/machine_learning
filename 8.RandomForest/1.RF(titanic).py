# [MXML-8-02]: 1.RF(titanic).py
# Implement Random Forest using MyDtreeClassifierRF.
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
from MyDTreeClassifierRF import MyDTreeClassifierRF
from sklearn.model_selection import train_test_split

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

n_estimators = 100
n_features = round(np.sqrt(x.shape[1])) # the number of features for column sampling
n_depth = 3                             # max_depth of tree

models = []  # base model list
for i in range(n_estimators):
    # Create a tree for Random Forest
    model = MyDTreeClassifierRF(max_depth=n_depth, 
                                max_samples = x_train.shape[0],
                                max_features=n_features)
    
    # train the tree.
    # subsampling by rows and columns is performed within the model
    model.fit(x_train, y_train)
    
    # save trained tree
    models.append(model)

# prediction
y_estimates = np.zeros(shape=(x_test.shape[0], n_estimators))
for i, model in enumerate(models):
    y_estimates[:, i] = model.predict(x_test)

# synthesizing the estimation results
y_prob = y_estimates.mean(axis=1)
y_pred = (y_prob >= 0.5) * 1
print('\nAccuracy = {:.4f}'.format((y_pred == y_test).mean()))

models
y_estimates.shape
y_estimates
y_estimates[0, :]
(y_estimates[0, :] == 0.0).sum()
(y_estimates[0, :] == 1.0).sum()
y_prob[0]
y_pred[0]
