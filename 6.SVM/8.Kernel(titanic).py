# [MXML-6-06] 8.Kernel(titanic).py
# Classify Titanic dataset using CVXOPT and SVC
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/-WVI6b19pag
# 
import numpy as np
import pandas as pd
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the Titanic data and perform some simple preprocessing.
df = pd.read_csv('data/titanic.csv')
df['Age'].fillna(df['Age'].mean(), inplace = True)  # Replace with average
df['Embarked'].fillna('N', inplace = True)          # Replace with 'N'
df['Sex'] = df['Sex'].factorize()[0]                # encoding
df['Embarked'] = df['Embarked'].factorize()[0]      # encoding
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#  Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked
# 0       0       3    0  22.0      1      0   7.2500         0
# 1       1       1    1  38.0      1      0  71.2833         1
# 2       1       3    1  26.0      0      0   7.9250         0
# 3       1       1    1  35.0      1      0  53.1000         0
# 4       0       3    0  35.0      0      0   8.0500         0

# Generate training and test data
y = np.array(df['Survived']).reshape(-1,1).astype('float') * 2 - 1
x = np.array(df.drop('Survived', axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Normalize the training and test data
x_mean = x_train.mean(axis=0).reshape(1, -1)
x_std = x_train.std(axis=0).reshape(1, -1)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# kernel function
def kernel(a, b, r=0.5):
    return np.exp(-r * np.linalg.norm(a - b)**2)
    
C = 1.0                # regularization constant
N = x_train.shape[0]   # the number of data points

# Kernel matrix. k(xi, xj) = φ(xi)φ(xj).
K = np.array([kernel(x_train[i], x_train[j]) 
        for i in range(N) 
        for j in range(N)]).reshape(N, N)
                  
# Construct the matrices required for QP in standard form.
H = np.outer(y_train, y_train) * K
P = cvxopt_matrix(H)
q = cvxopt_matrix(np.ones(N) * -1)
A = cvxopt_matrix(y_train.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

g = np.vstack([-np.eye(N), np.eye(N)])
G = cvxopt_matrix(g)

h1 = np.hstack([np.zeros(N), np.ones(N) * C])
h = cvxopt_matrix(h1)

# solver parameters
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Perform QP
sol = cvxopt_solvers.qp(P, q, G, h, A, b)

# the solution to the QP, λ
lamb = np.array(sol['x'])

# Find support vectors
sv_i = np.where(lamb > 1e-5)[0]
sv_m = lamb[sv_i]    # lambda
sv_x = x_train[sv_i]
sv_y = y_train[sv_i]

# Calculate b using the support vectors and calculate the average.
def cal_wphi(cond):
    wphi = []
    idx = np.where(cond)[0]
    for i in idx:
        wp = [sv_m[j] * sv_y[j] * kernel(sv_x[i], sv_x[j]) \
              for j in range(sv_x.shape[0])]
        wphi.append(np.sum(wp))
    return wphi

b = -(np.max(cal_wphi(sv_y > 0)) + np.min(cal_wphi(sv_y < 0))) / 2.

# Predict the class of test data.
n_test = x_test.shape[0]
n_sv = sv_x.shape[0]
ts_K = np.array([kernel(sv_x[i], x_test[j]) 
        for i in range(n_sv) 
        for j in range(n_test)]).reshape(n_sv, n_test)
        
# decision function
y_hat = np.sum(sv_m * sv_y * ts_K, axis=0).reshape(-1, 1) + b
y_pred = np.sign(y_hat)

acc = (y_pred == y_test).mean()
print('\nCVXOPT: The accuracy of the test data= {:.4f}'.format(acc))

# Compare with sklearn's SVC results.
from sklearn.svm import SVC
model = SVC(C=C, kernel='rbf', gamma=0.5)
model.fit(x_train, y_train.reshape(-1,))
y_pred = model.predict(x_test)

acc = (y_pred == y_test.reshape(-1,)).mean()
print('   SVC: The accuracy of the test data= {:.4f}'.format(acc))
