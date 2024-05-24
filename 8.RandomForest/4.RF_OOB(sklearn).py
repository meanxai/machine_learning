# [MXML-8-03] 4.RF_OOB(sklearn).py
# Add Out-Of-Bag (OOB) score feature to 2.RF(sklearn).py
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read preprocessed Titanic data.
df = pd.read_csv('data/titanic_clean.csv')
y = np.array(df['Survived'])
x = np.array(df.drop('Survived', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y)

N = x_train.shape[0]  # the number of train data points
n_estimators = 50
n_depth = 5                                      # tree의 max_depth
max_features = round(np.sqrt(x_train.shape[1]))  # for column sub sampling

# Implement Random Forest using DecisionTreeClassifier
# ----------------------------------------------------
# majority vote for iob_pred, or oob_pred
# p = iob_pred or oob_pred
def majority_vote(p):
    cnt_0 = (p == 0).sum(axis=1)
    cnt_1 = (p == 1).sum(axis=1)
    cnts = np.array([cnt_0, cnt_1])   # shape = (2, 668)
    return np.argmax(cnts, axis=0)

sk_models = []  # base model list
iob_score = []  # Error rate measured with IOB
oob_score = []  # Error rate measured with OOB

# initialize IOB and OOB prediction map
iob_pred = np.ones(shape=(N, n_estimators)) * -1
oob_pred = np.ones(shape=(N, n_estimators)) * -1
i_train = set(np.arange(0, N))

# Create n_estimators models
for i in range(n_estimators):
    # row subsampling
    i_row = np.random.choice(np.arange(0, N), N, replace=True)
    x_sample = x_train[i_row, :] # bootstrapped data
    y_sample = y_train[i_row]
           
    # Create a subtree for Random Forest
    # Column subsampling for each split is performed within the model.
    model = DecisionTreeClassifier(max_depth=n_depth,
                                   max_features="sqrt")
    model.fit(x_sample, y_sample)
    
    # save trained subtree
    sk_models.append(model)

    # Create IOB and OOB prediction map
    i_oob = list(i_train - set(i_row))   # OOB index
    iob_pred[i_row, i] = model.predict(x_train[i_row])
    oob_pred[i_oob, i] = model.predict(x_train[i_oob])
    
    # Calculate IOB and OOB score
    y_trn = majority_vote(iob_pred)
    y_oob = majority_vote(oob_pred)
    
    iob_score.append((y_trn != y_train).mean())
    oob_score.append((y_oob != y_train).mean())
    
# Visualize IOB and OOB score
plt.figure(figsize=(6, 4))
plt.plot(iob_score, color='blue', lw=3.0, label='IOB error')
plt.plot(oob_score, color='red', lw=3.0, label='OOB error')
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('OOB error rate')
plt.show()

# prediction
y_estimates = np.zeros(shape=(x_test.shape[0], n_estimators))
for i, model in enumerate(sk_models):
    y_estimates[:, i] = model.predict(x_test)

# synthesizing the estimation results
y_prob = y_estimates.mean(axis=1)
y_pred = (y_prob >= 0.5) * 1
accuracy = (y_pred == y_test).mean()
print('\nThe result from DecisionTreeClassifier:')
print('Accuracy of test data = {:.4f}'.format(accuracy))
print('Final OOB error rate = {:.4f}'.format(oob_score[-1]))

# OOB probability
# In theory, it would be 0.3679.
# This means that x_train is selected with probability 0.6321
# by row subsampling. (1.0 - 0.3679 = 0.6321)
oob_percent = ((oob_pred >= 0).sum(axis=0) / N).mean()
print('OOB probability = {:.4f}'.format(oob_percent))

# Implement Random Forest using RandomForestClassifier
# ----------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=n_depth,
                                  max_features="sqrt",  # default
                                  max_samples=N,        # default
                                  oob_score=True)
rf_model.fit(x_train, y_train)
y_pred1 = rf_model.predict(x_test)

print('\nThe result from RandomForestClassifier:')
print('Accuracy of test data = {:.4f}'.format((y_pred1 == y_test).mean()))
print('Final OOB error rate = {:.4f}'.format(1 - rf_model.oob_score_))
