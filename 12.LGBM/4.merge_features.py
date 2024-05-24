# [MXML-12-04] 4.merge_features.py
# Implementation of Algorithm 4: Merge Exclusive Features
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/orSRRtWtPwE
# 
import numpy as np

x = np.array([[1, 1, 0, 0, 1],
              [0, 0, 1, 1, 1],
              [1, 2, 0, 0, 2],
              [0, 0, 2, 3, 1],
              [2, 1, 0, 0, 3],
              [3, 3, 0, 0, 1],
              [0, 0, 3, 0, 2],
              [1, 2, 3, 4, 3],   # <-- conflict here
              [1, 0, 1, 0, 0],
              [2, 3, 0, 0, 2]])

# Algorithm 4: Merge Exclusive Features
def merge_features(numData, F):
    binRanges = [0]
    totalBin = 0
    for f in F:
        totalBin += np.max(f)
        binRanges.append(totalBin)

    newBin = np.zeros(numData, dtype=int)
    for i in range(numData):
        newBin[i] = 0
        for j in range(len(F)):
            if F[j][i] != 0:
                newBin[i] = F[j][i] + binRanges[j]
    return newBin, binRanges

# modified Algorithm 4 (skip-zero-version)
def merge_features2(numData, F):
    binRanges = [0]
    totalBin = 0
    for f in F:
        totalBin += np.max(f)
        binRanges.append(totalBin)

    # initialize newBin with F[0] to skip zero in binRanges[0]
    newBin = F[0]
    for i in range(numData):
        for j in range(1, len(F)):
            if F[j][i] != 0:
                newBin[i] = F[j][i] + binRanges[j]
    return newBin, binRanges

bundles = [[4], [0, 3], [1, 2]] # The result of Greedy Bundling

F = [x[:, i] for i in bundles[1]]
newBin, binRanges = merge_features(x.shape[0], F)
print('\nnewBin:', newBin)
print('binRanges:', binRanges)

newBin, binRanges = merge_features2(x.shape[0], F)
print('\nnewBin:', newBin)
print('binRanges:', binRanges)
