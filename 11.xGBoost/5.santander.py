# [MXML-11-09] 5.santander.py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/fALcIVr6zjY
# 
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the Santander Customer Satisfaction Dataset.
# df.shape = (76020, 371)
df = pd.read_csv("data/santander.csv", encoding='latin-1')

# Replace the values of the 'var3' feature containing -99999999 with 2 
# and drop the 'ID' feature.
df['var3'].replace(-999999, 2, inplace=True)
df.drop('ID', axis = 1, inplace=True)

# Separate features and label from the dataset 
# and generate training and test data.
x = df.drop('TARGET', axis=1)
y = df['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x, y)

TREES = 200  # the number of trees
DEPTH = 5    # the depth of tree
ETA = 0.1    # learning rate, eta
LAMB = 1.0   # regularization constant
GAMMA = 0.1  # pruning constant
EPS = 0.03   # epsilon for approximate and weighted quantile sketch

# Create an XGBoost classification model and fit it to the training data
model = XGBClassifier(n_estimators = TREES,
                      max_depth = DEPTH,
                      learning_rate = ETA,    # η
                      gamma = GAMMA,          # γ for pruning
                      reg_lambda = LAMB,      # λ for regularization
                      base_score = 0.5,       # initial prediction value
                      missing = 0.0,          # for sparsity-aware
                      subsample = 0.5,        # Subsample ratio of the training instance
                      colsample_bynode = 0.5, # Subsample ratio of columns for each split
                      max_bin = int(1/EPS),   # sketch_eps is replaced by max_bin
                      tree_method = 'approx') # weighted quantile sketch

model.fit(x_train, y_train)

# Predict the test data and measure the performance with ROC-AUC.
y_prob = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print('\nROC-AUC = {:.4f}'.format(auc))

# colsample_bytree (Optional[float]) – Subsample ratio of columns when constructing 
#                                      each tree.
# colsample_bylevel (Optional[float]) – Subsample ratio of columns for each level.

# colsample_bynode (Optional[float]) – Subsample ratio of columns for each split.

