# [MXML-6-07] 9.multiclass(OvO).py
# Implement multiclass classification of SVM by One-vs-One (OvO)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/MAde_oEYB-g
# 
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from itertools import combinations

# Generate the data with 4 clusters.
x, y = make_blobs(n_samples=400, n_features=2, 
                centers=[[0., 0.], [0.5, 0.5], [1., 0.], [-0.8, 0.]], 
                cluster_std=0.17)

# Linear SVM
C = 1.0
model = SVC(C=C, kernel='linear', decision_function_shape='ovo')
model.fit(x, y)

w = model.coef_
b = model.intercept_
print("w:\n ", w.round(3))     # shape=(6,2)
print("\nb:\n ", b.round(3))   # shape=(6,)

# Visualize the data and six boundaries.
plt.figure(figsize=(8,7))
colors = ['red', 'blue', 'green', 'black']
y_color= [colors[a] for a in y]
for label in model.classes_:
    idx = np.where(y == label)
    plt.scatter(x[idx, 0], x[idx, 1], s=100, c=colors[label], 
                alpha=0.5, label='class_' + str(label))

# Visualize six boundaries.
comb = list(combinations(model.classes_, 2))
x1_dec = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
for i in range(w.shape[0]):
    x2_dec = -(w[i, 0] * x1_dec + b[i]) / w[i, 1]
    plt.plot(x1_dec, x2_dec, label=str(comb[i]))
plt.xlim(-1.5, 1.8)    
plt.ylim(-0.7, 1.)
plt.legend()
plt.show()

# Predict the classes of the test data.
x_test = np.random.uniform(-1.5, 1.5, (2000, 2))
y_pred1 = model.predict(x_test)

# To understand how OvO works, let's manually implement the 
# process of model.predict(x_test). df.shape = (2000, 6)
df = np.dot(x_test, w.T) + b            # decision function
# df = model.decision_function(x_test)  # same as above

classes = model.classes_
n_class = classes.shape[0]

# Reference: https://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict
y_pred = []
for i in range(df.shape[0]):
    votes = np.zeros(n_class)
    for j in range(df.shape[1]):    # the number of boundaries
        # if df(i, j) > 0, then class=i, else class=j
        if df[i][j] > 0:            
            votes[comb[j][0]] += 1
        else:
            votes[comb[j][1]] += 1
        
    v = np.argmax(votes)            # majority vote
    y_pred.append(classes[v])
y_pred2 = np.array(y_pred)

# Compare the results of y_pred1 and y_pred2.
if (y_pred1 != y_pred2).sum() == 0:
    print("# y_pred1 and y_pred2 are exactly the same.")
else:
    print("# y_pred1 and y_pred2 are not the same.")

# Visualize test data and y_pred1
plt.figure(figsize=(8,7))
y_color= [colors[a] for a in y_pred1]
for label in model.classes_:
    idx = np.where(y_pred1 == label)
    plt.scatter(x_test[idx, 0], x_test[idx, 1], s=100, c=colors[label], 
                alpha=0.3, label='class_' + str(label))

plt.xlim(-1.5, 1.8)    
plt.ylim(-0.7, 1.)
plt.show()

