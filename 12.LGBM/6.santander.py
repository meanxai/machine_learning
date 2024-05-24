# [MXML-12-05] 6.santander.py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/NqpkYja5g2Y
# 
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Read the Santander Customer Satisfaction Dataset.
# df.shape = (76020, 371)
df = pd.read_csv("data/santander.csv", encoding='latin-1')

# Replace the values of the 'var3' feature containing -99999999 
# with 2 and drop the 'ID' feature.
df['var3'].replace(-999999, 2, inplace=True)
df.drop('ID', axis = 1, inplace=True)

# Separate features and label from the dataset 
# and generate training and test data.
x_feat = df.drop('TARGET', axis=1)
y_target = df['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x_feat, y_target)

# 1. XGBoost
# Create an XGBoost classification model and fit it to the training data
start_time = time.time()
model = XGBClassifier(n_estimators = 200,
                      max_depth = 5,
                      learning_rate = 0.1,    # η
                      gamma = 0.1,            # γ for pruning
                      reg_lambda = 1.0,       # λ for regularization
                      base_score = 0.5,       # initial prediction value
                      subsample = 0.5,        # Subsample ratio of the training instance
                      colsample_bynode = 0.5, # Subsample ratio of columns for each split
                      max_bin = int(1/0.03),  # sketch_eps is replaced by max_bin
                      tree_method = 'approx') # weighted quantile sketch

model.fit(x_train, y_train)

# Predict the test data and measure the performance with ROC-AUC.
y_prob = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print('\nXGBoost results:')
print('running time = {:.2f} seconds'.format(time.time() - start_time))
print('ROC-AUC = {:.4f}'.format(auc))

# 2. LightGBM
# Create a LightGBM model
start_time = time.time()
model = LGBMClassifier(n_estimators = 200,
                       max_depth = 5,
                       learning_rate = 0.1,
                       boosting_type="goss",  # default: gbdt - traditional gradient based decision tree
                       top_rate=0.3, 
                       other_rate=0.2,
                       enable_bundle=True,    # default: True. enable EFB
                       is_unbalance = True)

# training
model.fit(x_train, y_train)

# Predict the test data and measure the performance with AUC.
y_pred = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

print('\nLightGBM results:')
print('running time = {:.2f} seconds'.format(time.time() - start_time))
print("ROC AUC = {0:.4f}".format(auc))

# Draw the ROC curve
fprs, tprs, thresholds = roc_curve(y_test, y_pred)

plt.plot(fprs, tprs, label = 'ROC')
plt.plot([0,1], [0,1], '--', label = 'Random')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

