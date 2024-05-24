# [MXML-12-03] 3.greedy_bundling.py
# Algorithm 3: Greedy Bundling
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/Y-IvfsjmqOQ
# 
import numpy as np

x = np.array([[1, 1, 0, 0, 1],
              [0, 0, 1, 1, 1],
              [1, 2, 0, 0, 2],
              [0, 0, 2, 3, 1],
              [2, 1, 0, 0, 3],
              [3, 3, 0, 0, 1],
              [0, 0, 3, 0, 2],
              [1, 2, 3, 4, 3],
              [1, 0, 1, 0, 0],
              [2, 3, 0, 0, 2]])

# Create a conflict count matrix
n_row = x.shape[0]
n_col = x.shape[1]
conflictCnt = np.zeros((n_col, n_col))

for i in range(n_col):
    for j in range(i+1, n_col):
        # Count the number of conflicts.
        conflictCnt[i, j] = len(np.where(x[:, i] * x[:, j] > 0)[0])

# Copy upper triangle to lower triangle
# iu = (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), 
#       array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4]))
iu = np.triu_indices(n_col, 1)
il = (iu[1], iu[0])
conflictCnt[il] = conflictCnt[iu]

# Create a search order matrix
degree = conflictCnt.sum(axis=0)
searchOrder = np.argsort(degree)[::-1]  # descending order

# ----------------------------
# Algorithm 3: Greedy Bundling
# ----------------------------
K = 1        # max conflict count
bundles = []
bundlesConflict = []
for i in searchOrder:  # i = [4, 0, 1, 2, 3]
    needNew = True
    for j in range(len(bundles)):
        cnt = conflictCnt[bundles[j][-1], i]
        # Only edges less than or equal to K are considered.
        if cnt + bundlesConflict[j] <= K:
            # Add the feature number i to the j-th bundle.
            bundles[j].append(i)
            
            # Update the number of conflicts of features in the 
            # j-th bundle.
            bundlesConflict[j] += cnt     
            needNew = False
            break
        
    if needNew:
        bundles.append([i])
        bundlesConflict.append(0.)

print('\nconflictCnt:\n', conflictCnt)
print('\nsearchOrder:\n', searchOrder)

print('\nbundles:', bundles)
print('bundlesConflict:', bundlesConflict)

# conflictCnt:
#    0   1   2   3   4
# 0 [0., 6., 2., 1., 6.]
# 1 [6., 0., 1., 1., 6.]
# 2 [2., 1., 0., 3., 4.]
# 3 [1., 1., 3., 0., 3.]
# 4 [6., 6., 4., 3., 0.]

# searchOrder
# array([4, 0, 1, 2, 3])
#
# bundles:
#   j=0   j=1     j=2  ← bundle number
# +--↓-----↓-------↓-----+
# | [4]  [0, 3]  [1, 2]  |
# +----------------------+
#
# bundlesConflict
# +----------------------+
# |  0      1       1    |
# +----------------------+
