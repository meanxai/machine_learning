# [MXML-12-05] 5.efb_onehot.py
# Merge one-hot encoded features using EFB
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/NqpkYja5g2Y
# 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Algorithm 3: Greedy Bundling algorithm
def greedy_bundling(x, K):
    # Create a conflict count matrix
    n_row = x.shape[0]
    n_col = x.shape[1]
    conflictCnt = np.zeros((n_col, n_col))
    
    for i in range(n_col):
        for j in range(i+1, n_col):
            # Count the number of conflicts.
            conflictCnt[i, j] = len(np.where(x[:, i] * x[:, j] > 0)[0])
    
    # Copy upper triangle to lower triangle
    iu = np.triu_indices(n_col, 1)
    il = (iu[1], iu[0])
    conflictCnt[il] = conflictCnt[iu]
    
    # Create a search order matrix
    degree = conflictCnt.sum(axis=0)
    searchOrder = np.argsort(degree)[::-1]  # descending order
    
    bundles = []
    bundlesConflict = []
    for i in searchOrder:
        needNew = True
        for j in range(len(bundles)):
            cnt = conflictCnt[bundles[j][-1], i]
            if cnt + bundlesConflict[j] <= K:
                bundles[j].append(i)
                bundlesConflict[j] += cnt     
                needNew = False
                break
            
        if needNew:
            bundles.append([i])
            bundlesConflict.append(0.)
    return bundles

# Algorithm 4: Merge Exclusive Features (skip-zero-version)
def merge_features(numData, F):
    binRanges = [0]
    totalBin = 0
    for f in F:
        totalBin += np.max(f)
        binRanges.append(totalBin)

    newBin = F[0]  # initialize newBin to F[0]
    for i in range(numData):
        for j in range(1, len(F)):
            if F[j][i] != 0:
                newBin[i] = F[j][i] + binRanges[j]
    return newBin, binRanges

# Generate random data and perform one-hot encoding.
n_samples = 100
n_features = 4
x = np.random.randint(low=0, high=4, size=(n_samples, n_features))
enc = OneHotEncoder()
x_ohe = enc.fit_transform(x).toarray()

print('Original features [:5]:'); print(x[:5])
print('\nOne-hot encoding [:5]:'); print(x_ohe[:5])

# Find bundles
bundles = greedy_bundling(x_ohe, K=1)

# If we know the bundles exactly, like this,
# bundles = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
# we can get the original features from the merged features.

print('\nbundles:', bundles)
# [[14, 12, 15, 13], [10, 8, 11, 9], [5, 4, 6, 7], [3, 2, 1, 0]]

# Merge one-hot encoded features
x_efb = np.zeros(shape=x.shape).astype('int')
for i, bundle in enumerate(bundles):
    F = [x_ohe[:, i] for i in bundle]
    newBin, binRanges = merge_features(x_ohe.shape[0], F)
    x_efb[:, i] = np.array(newBin) - 1

print('\nOriginal features [:5]:'); print(x[:5])
print('\nMerged features [:5]:'); print(x_efb[:5])

