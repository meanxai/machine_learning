# [MXML-2-11] MyDTreeRegressor.py
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/Bc-k9Dv5SNg
#
import numpy as np
from collections import Counter
import copy

# Implement the Decision Tree Regressor using binary tree.
class MyDTreeRegressor:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.estimator1 = dict() # tree result-1: each leaf contains the data index.
        self.estimator2 = dict() # tree result-2: each leaf contains average value.
        self.feature = None      # feature value. will be x_train when fit() is called.
        self.target = None       # target value. will be y_train when fit() is called.
    
    # Split a node into left and right node.
    # Find the point with the smallest MSE and split the node at that point.
    # did: data index on the leaf node.
    def node_split(self, did):
        n = did.shape[0]
        
        # Split the node into all candidates for all features and find
        # the best feature and the best split point with the smallest MSE. 
        # fid: feature_id
        min_mse = 999999
        for fid in range(self.feature.shape[1]):
            # feature data to be split
            x_feat = self.feature[did, fid].copy()
            
            # split x_feat using the best feature and the best split point.
            # Note: The code below is inefficient because it sorts x_feat 
            #       every time it is split. Future improvements are needed.
            
            # remove duplicates of x_feat and sort in ascending order
            x_uniq = np.unique(x_feat)
            
            # list up all the candidates, which are the midpoints of adjacent data.            
            s_point = [np.mean([x_uniq[i-1], x_uniq[i]]) for i in range(1, len(x_uniq))]
            
            # len(s_point) > 1: Calculate MSE for all candidates, 
            #                   and find the candidate with the smallest MSE.
            # len(s_point) < 1: skip the for-loop. x_feat either has only one data 
            #                   or all have the same value. No need to split.            
            for p in s_point:
                # split x_feat into the left and the right node.
                left = did[np.where(x_feat <= p)[0]]
                right = did[np.where(x_feat > p)[0]]
                               
                # calculate MSE after splitting. MSE is the same as variance in this case.
                l_mse = self.target[left].var()
                r_mse = self.target[right].var()
                mse = l_mse * (left.shape[0] / n) + r_mse * (right.shape[0] / n)
                
                # find where the MSE is smallest.
                if mse < min_mse:
                    min_mse = mse
                    b_fid = fid      # best feature id
                    b_point = p      # best split point
                    b_left = left    # data index on the left node.
                    b_right = right  # data index on the right node.
        
        if min_mse < 999999.:   # split
            return {'fid':b_fid, 'split_point':b_point, 'left':b_left, 'right':b_right}
        else:
            return  None        # No split

    # Create a binary tree using recursion
    def recursive_split(self, node, curr_depth):
        left = node['left']
        right = node['right']
        
        # exit recursion
        if curr_depth >= self.max_depth:
            return
        
        # process recursion
        s = self.node_split(left)
        if isinstance(s, dict):   # split to the left done.
            node['left'] = s
            self.recursive_split(node['left'], curr_depth+1)
    
        s = self.node_split(right)
        if isinstance(s, dict):   # split to the right done.
            node['right'] = s
            self.recursive_split(node['right'], curr_depth+1)
     
    # Change the data in the leaf node to average value.
    def update_leaf(self, d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == 'left' or key == 'right':
                    rtn = self.update_leaf(value)
                    if rtn[0] == 1:      # leaf node
                        d[key] = rtn[1]
            return 0, 0  # the first 0 indicates this is not a leaf node.
        else:            # leaf node
            # the first 1 indicates this is a leaf node.
            return 1, self.target[d].mean()
    
    # create a tree using training data, and return the result of the tree.
    # x : feature data, y: target data
    def fit(self, x, y):
        self.feature = x
        self.target = y
        
        # Initially, the root node holds all data indices.
        root = self.node_split(np.arange(x.shape[0]))
        if isinstance(root, dict):
            self.recursive_split(root, curr_depth=1)
        
        # tree result-1. Every leaf node has data indices.
        self.estimator1 = root
        
        # tree result-2. Every leaf node has average value.
        self.estimator2 = copy.deepcopy(self.estimator1)
        self.update_leaf(self.estimator2)             # tree result-2
        return self.estimator2

    # Estimate the target value of a test data.
    def x_predict(self, p, x):
        if x[p['fid']] <= p['split_point']:
            if isinstance(p['left'], dict):          # recursion if not leaf.
                return self.x_predict(p['left'], x)  # recursion
            else:                                    # return the value in the leaf, if leaf.
                return p['left']
        else:
            if isinstance(p['right'], dict):         # recursion if not leaf.
                return self.x_predict(p['right'], x) # recursion
            else:                                    # return the value in the leaf, if leaf.
                return p['right']

    # Estimate the target class of x_test.
    def predict(self, x_test):
        p = self.estimator2    # predictor
        y_pred = [self.x_predict(p, x) for x in x_test]
        return np.array(y_pred)
