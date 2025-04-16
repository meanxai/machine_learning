# [MXML-1-04] 5.WKNN.py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/Lu6GAc4FYz8
#
import numpy as np

# Let's assume that the distance matrix between the test data and 
# the training data is given as follows. shape = (5, 10)
dist = np.array(
# train: 0     1    2    3    4    5    6    7    8    9       test
       [[5. , 3.5, 4.3, 3.4, 1.4, 6.5, 2.7, 5.1, 2.9, 2.8],   # i=0
        [4.4, 1.9, 3.6, 3.3, 0.5, 5.5, 2.1, 4.4, 1.3, 2.3],   # i=1
        [4.6, 1. , 3.9, 4.4, 3. , 4.7, 3.2, 4.4, 1.4, 3.5],   # i=2
        [4.7, 0.6, 3.9, 4.1, 1.7, 5.3, 2.7, 4.6, 0.4, 3. ],   # i=3
        [3. , 3.6, 2.4, 1.4, 2.4, 4.8, 1.2, 3.2, 3. , 1.1]])  # i=4

# target class y   (0  1  2  3  4  5  6  7  8  9)
y_train = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
C = [0, 1]   # the class y is either 0 or 1
K = 7        # 7-nearest neighbors
T = 5        # the number of test data points

# Find K nearest neighbors
i_near = np.argsort(dist, axis=1)[:, :K]
y_near = y_train[i_near]

# Compute the inverse distance
w_dist = np.array([dist[i, :][i_near[i, :]] for i in range(T)])
w_inv = 1. / w_dist

# Predict the class of test data using the inverse weighted distance
y_pred = []
for i in range(T):
    iw_dist = [w_inv[i][y_near[i] == j].sum() for j in C]
    y_pred.append(np.argmax(iw_dist / w_inv[i].sum()))

print(y_pred)
