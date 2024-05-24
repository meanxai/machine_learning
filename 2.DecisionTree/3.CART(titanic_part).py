# [MXML-2-08] 3.CART(sklearn).py
# DecisionTreeClassifier in sklearn
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/XqNuY1RHlNU
#
# The characteristics of DecisionTreeClassifier:
# 1. Use the CART algorithm (binary tree). 
#    ID3/C4.5 (general tree) is not supported. 
# 2. Categorical feature is not directly supported.
#    All categorical features (e.g. 'female', 'male') must be 
#    converted to numeric data (e.g. 0, 1).
#    All numeric features are treated as continuous features.
#    Split using inequality. (e.g. sex ≤ 0.5)
# -----------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Of the Titanic dataset, only the following three features are used.
feat_names = ['Pclass', 'Sex', 'Age']
df = pd.read_csv('data/titanic.csv')[feat_names + ['Survived']]
df['Sex'] = df['Sex'].factorize()[0] # convert string to number
df = df.dropna()                     # Delete all rows with missing values.
col_names = list(df.columns)

# Separate the Titanic data into features and target class.
x_data = np.array(df[feat_names])  # features
y_data = np.array(df['Survived'])  # target class

# Split the data into training, validation and test data.
x_train, x_test, y_train, y_test = \
    train_test_split(x_data, y_data, test_size = 0.3)

x_test, x_eval, y_test, y_eval = \
    train_test_split(x_test, y_test, test_size = 0.5)
    
# Create decision tree models of various depths, 
# and measure the accuracy of validation data for each model.
train_acc = []
eval_acc = []
max_depth = 8
for d in range(1, max_depth+1):
    model = DecisionTreeClassifier(max_depth=d)
    model.fit(x_train, y_train)
    
    # Measure the accuracy of this model using the training data.
    y_pred = model.predict(x_train)
    train_acc.append((y_pred == y_train).mean())

    # Measure the accuracy of this model using the validation data.
    y_pred = model.predict(x_eval)
    eval_acc.append((y_pred == y_eval).mean())
    print('Depth = {}, train_acc = {:.4f}, eval_acc = {:.4f}'\
          .format(d, train_acc[-1], eval_acc[-1]))

# Find the optimal depth with the highest accuracy of validation data.
opt_depth = np.argmax(eval_acc) + 1

# Visualize accuracy changes as depth changes.
plt.plot(train_acc, marker='o', label='train')
plt.plot(eval_acc, marker='o', label='evaluation')
plt.legend()
plt.title('Accuracy')
plt.xlabel('tree depth')
plt.ylabel('accuracy')
plt.xticks(np.arange(max_depth), np.arange(1, max_depth+1))
plt.axvline(x=opt_depth-1, ls='--')
plt.ylim(0.5, 1.0)
plt.show()

# Regenerate the tree with optimal depth. 
# model = DecisionTreeClassifier(max_depth=opt_depth)

# I set max_step=3 as a constant value for tree visualization.
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train, y_train)

# Use test data to evaluate final performance.
y_pred = model.predict(x_test)
test_acc = (y_pred == y_test).mean()
print('Optimal depth = {}, test_acc = {:.4f}'.format(opt_depth, test_acc))
        
# Visualize the tree
# plt.figure(figsize=(20,10))
plt.figure(figsize=(14,6))
tree.plot_tree(model, feature_names = feat_names, fontsize=10)
plt.show()

# Analyze the importance of features.
feature_importance = model.feature_importances_
n_feature = x_train.shape[1]
idx = np.arange(n_feature)

plt.barh(idx, feature_importance, align='center', color='green')
plt.yticks(idx, col_names[:-1], size=12)
plt.xlabel('importance', size=15)
plt.ylabel('feature', size=15)
plt.show()

print('feature importance = {}'\
      .format(feature_importance.round(3)))
