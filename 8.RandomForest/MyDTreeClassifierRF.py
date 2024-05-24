# [MXML-8-02] MySubtreeRF.py: Create a subtree for Random Forest.
# This code is an upgrade version of MyDTreeClassifier 
# shown in [MXML-2-07] video.
# Features: subsampling by rows and columns,
#           predict Out-Of-Bag (OOB) data points
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/9si5fELmtg0
# 
import numpy as np
from collections import Counter
import copy

# Implement a simplified Random Forest using binary tree.
class MyDTreeClassifierRF:
    def __init__(self, max_depth, max_samples, max_features):
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.max_features = max_features
        self.u_class = None      # unique class (target y value)
        self.estimator1 = dict() # tree result-1: each leaf contains the data index.
        self.estimator2 = dict() # tree result-2: each leaf contains the majority class.
        self.feature = None      # feature value. will be x_train when fit() is called.
        self.target = None       # target value. will be y_train when fit() is called.
        self.iob_pred = None     # predicted classes of train data points
        self.oob_pred = None     # predicted classes of OOB data points
           
    # Calculate Gini index of a leaf node
    def gini_index(self, leaf):
        n = leaf.shape[0]
        
        if n > 1:
            gini = 1.0
            for c in self.u_class:
                cnt = (self.target[leaf] == c).sum()
                gini -= (cnt / n) ** 2
            return gini
        else:
            return 0.0

    # split a node into left and right.
    # Find the best split point with highest information gain, and split node with it.
    # did: data index on the leaf node.
    def node_split(self, did):
        n = did.shape[0]
        
        # Gini index of parent node before splitting.
        p_gini = self.gini_index(did)
        
        # perform column subsampling without replacement
        m = self.max_features
        p = self.feature.shape[1]
        f_list = np.random.choice(np.arange(0, p), m, replace=False)
        
        # Split the node into all candidates for all features and find
        # the best feature and the best split point with the highest 
        # information gain.
        # fid: feature_id
        max_ig = -999999
        for fid in f_list:
            # feature data to be split
            x_feat = self.feature[did, fid].copy()
            
            # split x_feat using the best feature and the best split point.
            # Note: The code below is inefficient because it sorts x_feat 
            #       every time it is split. Future improvements are needed.
            
            # remove duplicates of x_feat and sort in ascending order
            x_uniq = np.unique(x_feat)
            
            # list up all the candidates, which are the midpoints of adjacent data points.
            s_point = [np.mean([x_uniq[i-1], x_uniq[i]]) for i in range(1, len(x_uniq))]
            
            # len(s_point) > 1: Calculate the information gain for all candidates, 
            #                   and find the candidate with the largest information gain.
            # len(s_point) < 1: skip the for-loop. x_feat either has only one data point
            #                   or has all the same values. No need to split.
            for p in s_point:
                # split x_feat into the left and the right node.
                left = did[np.where(x_feat <= p)[0]]
                right = did[np.where(x_feat > p)[0]]
                               
                # calculate Gini index after splitting.
                l_gini = self.gini_index(left)
                r_gini = self.gini_index(right)
                
                # calculate information gain
                ig = p_gini - (l_gini * left.shape[0] / n) - (r_gini * right.shape[0] / n)

                # find where the information gain is greatest.
                if ig > max_ig:
                    max_ig = ig
                    b_fid = fid      # best feature id
                    b_point = p      # best split point
                    b_left = left    # data index on the left node.
                    b_right = right  # data index on the right node.
        
        if max_ig > 0.:     # split
            return {'fid':b_fid, 'split_point':b_point, 'left':b_left, 'right':b_right}
        else:
            return  None    # No split

    # Create a binary tree using recursion
    def recursive_split(self, node, curr_depth):
        left = node['left']
        right = node['right']
        
        # exit recursion
        if curr_depth >= self.max_depth:
            return
        
        # recursion
        s = self.node_split(left)
        if isinstance(s, dict):   # splitting to the left done.
            node['left'] = s
            self.recursive_split(node['left'], curr_depth+1)
    
        s = self.node_split(right)
        if isinstance(s, dict):   # splitting to the right done.
            node['right'] = s
            self.recursive_split(node['right'], curr_depth+1)
           
    # majority vote
    def majority_vote(self, did):
        c = Counter(self.target[did])
        return c.most_common(1)[0][0]
    
    # Change the data in the leaf node to majority class.
    def update_leaf(self, d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == 'left' or key == 'right':
                    rtn = self.update_leaf(value)
                    if rtn[0] == 1:      # leaf node
                        d[key] = rtn[1]
            return 0, 0  # the first 0 means this is not a leaf node.
        else:            # leaf node
            # the first 1 means this is a leaf node.
            return 1, self.majority_vote(d)
    
    # create a tree using training data, and return the result of the tree.
    # x : feature data, y: target data
    def fit(self, x, y):
        # perform row subsampling with replacement
        n = x.shape[0]
        i_rows = np.random.choice(np.arange(0, n), 
                                  self.max_samples, 
                                  replace=True)
        
        self.feature = x[i_rows, :]
        self.target = y[i_rows]
        self.u_class = np.unique(y)
        
        # Initially, the root node holds all data points IDs.
        root = self.node_split(np.arange(x.shape[0]))
        if isinstance(root, dict):
            self.recursive_split(root, curr_depth=1)
        
        # tree result-1. Every leaf node has data point IDs.
        # It is used for predict_proba(), etc.
        self.estimator1 = root
        
        # tree result-2. Every leaf node has the majority class.
        # It is used for predict().
        self.estimator2 = copy.deepcopy(self.estimator1)
        self.update_leaf(self.estimator2)             # tree result-2
        
        # predict Out-Of-Bag (OOB) data points
        # predict In-Of-Bag (IOB) and Out-Of-Bag (OOB) data points
        # initialize the predicted classes of IOB and OOB data points
        self.iob_pred = np.ones(shape=(x.shape[0],), dtype=int) * -1
        self.oob_pred = np.ones(shape=(x.shape[0],), dtype=int) * -1
        
        # predict training dataset
        y_pred = self.predict(x)
        
        # predict IOB and OOB data points
        i_train = set(np.arange(0, x.shape[0]))
        i_oobs = list(i_train - set(i_rows))   # OOB data point IDs
        
        self.iob_pred[i_rows] = y_pred[i_rows]   # for IOB data
        self.oob_pred[i_oobs] = y_pred[i_oobs]   # for OOB data

        return self.iob_pred, self.oob_pred

    # Estimate the target class of a test data.
    def x_predict(self, p, x):
        if x[p['fid']] <= p['split_point']:
            if isinstance(p['left'], dict):           # recursion if not leaf.
                return self.x_predict(p['left'], x)   # recursion
            else:                                     # return the value in the leaf, if leaf.
                return p['left']
        else:
            if isinstance(p['right'], dict):          # recursion if not leaf.
                return self.x_predict(p['right'], x)  # recursion
            else:                                     # return the value in the leaf, if leaf.
                return p['right']

    # Estimate the target class of a x_test.
    def predict(self, x_test):
        p = self.estimator2    # predictor
        y_pred = [self.x_predict(p, x) for x in x_test]
        return np.array(y_pred)
