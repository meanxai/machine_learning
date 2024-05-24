# [MXML-3-03] 3.boston(sklearn).py
# prediction of Boston house price
# using sklear’s LinearRegression, Ridge, Lasso
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/gLekbL_pI1A
#
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import pickle

# Read Boston house price dataset
with open('data/boston_house.pkl', 'rb') as f:
    data = pickle.load(f)

x = data['data']      # features, shape = (506, 13)
y = data['target']    # target, shape = (506,)

# Split the dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 1. LinearRegression()
# ---------------------
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Visually check the predicted and actual y values ​​of the test data.
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, s=20, c='r')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

# 평가용 데이터의 R2를 확인한다.
r2 = model.score(x_test, y_test)
print('\nR2 (LinearRegression) = {:.3f}'.format(r2))

# 2. Ridge regularization
# -----------------------
model = Ridge(alpha=0.01)
model.fit(x_train, y_train)
r2 = model.score(x_test, y_test)
print('R2 (Ridge) = {:.3f}'.format(r2))

# 3. Lasso regularization
# -----------------------
model = Lasso(alpha=0.01)
model.fit(x_train, y_train)
r2 = model.score(x_test, y_test)
print('R2 (Lasso) = {:.3f}'.format(r2))
