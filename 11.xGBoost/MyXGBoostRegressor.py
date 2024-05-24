# [MXML-11-03] MyXGBoostRegressor.py
# Upgraded version of CART in [MXML-02-07] [MXML-02-11] for XGBoost
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/Ms_xxQFrTWc
# 
import numpy as np
from collections import Counter
import copy

# Implement "Exact Greedy Algorithm for Split Finding" presented
# in XGBoost paper. [1] Tianqi Chen et, al., 2016, XGBoost: A Scalable Tree Boosting
class MyXGBRegressionTree:
    def __init__(self, max_depth, reg_lambda, prune_gamma):
        self.max_depth = max_depth      # depth of the tree
        self.reg_lambda = reg_lambda    # regularization constant
        self.prune_gamma = prune_gamma  # pruning constant
        self.estimator1 = None          # tree result-1
        self.estimator2 = None          # tree result-2
        self.feature = None             # feature x.
        self.residual = None            # residuals
        self.base_score = None          # initial prediction 

    # [1] 2.1 Regularized Learning Objective
    # Algorithm 1: Exact Greedy Algorithm for Split Finding
    # Split a node into left and right. Find the best split point 
    # with highest gain and split the node with the point.
    def node_split(self, did):
        r = self.reg_lambda
        max_gain = -np.inf
        d = self.feature.shape[1]     # feature dimension
        G = -self.residual[did].sum() # G before split
        H = did.shape[0]              # the number of residuals
        p_score = (G ** 2) / (H + r)  # score before the split
        
        for k in range(d):
            GL = HL = 0.0
            
            # split x_feat using the best feature and the best 
            # split point. The code below is inefficient because 
            # it sorts x_feat every time it is split. 
            # Future improvements are needed.  
            x_feat = self.feature[did, k]
            
            # remove duplicates of x_feat and sort in ascending order
            x_uniq = np.unique(x_feat)
            s_point = [np.mean([x_uniq[i-1], x_uniq[i]]) for i in range(1, len(x_uniq))]
            l_bound = -np.inf            # lower left bound
            
            for j in s_point:
                # split x_feat into the left and the right node.
                left = did[np.where(np.logical_and(x_feat > l_bound, x_feat <= j))[0]]
                right = did[np.where(x_feat > j)[0]]
                
                # Calculate the scores after splitting
                GL -= self.residual[left].sum()
                HL += left.shape[0]
                GR = G - GL
                HR = H - HL
                
                # Calculate gain for this split
                gain = (GL ** 2)/(HL + r) + (GR ** 2)/(HR + r) - p_score
                                             
                # find the point where the gain is greatest.
                if gain > max_gain:
                    max_gain = gain
                    b_fid = k        # best feature id
                    b_point = j      # best split point
                l_bound = j
                
        if max_gain >= self.prune_gamma:
            # split the node using the best split point
            x_feat = self.feature[did, b_fid]
            b_left = did[np.where(x_feat <= b_point)[0]]
            b_right = did[np.where(x_feat > b_point)[0]]
            return {'fid':b_fid, 'split_point':b_point, 'gain':max_gain, 'left':b_left, 'right':b_right}
        else:
            return  np.nan    # no split

    # Create a binary tree using recursion
    def recursive_split(self, node, curr_depth):
        left = node['left']
        right = node['right']
        
        # exit recursion
        if curr_depth >= self.max_depth:
            return
        
        # process recursion
        s = self.node_split(left)
        if isinstance(s, dict):   # split
            node['left'] = s
            self.recursive_split(node['left'], curr_depth+1)
    
        s = self.node_split(right)
        if isinstance(s, dict):   # split
            node['right'] = s
            self.recursive_split(node['right'], curr_depth+1)

    # Calculate the output value of a leaf node
    def output_value(self, did):
        r = self.residual[did]
        return np.sum(r) / (did.shape[0] + self.reg_lambda)
    
    # Calculate output values for every leaf node in a tree
    def output_leaf(self, d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == 'left' or key == 'right':
                    rtn = self.output_leaf(value)
                    if rtn[0] == 1:      # leaf node
                        d[key] = rtn[1]
            return 0, 0  # first 0 = non-leaf node
        else:            # leaf node
            return 1, self.output_value(d) # # first 1 = leaf node
    
    # It creates a tree using the training data, and returns the 
    # result of the tree. (x : feature data, y: residuals)
    def fit(self, x, y):
        self.feature = x
        self.residual = y
        self.base_score = y.mean()  # initial prediction
        
        root = self.node_split(np.arange(x.shape[0]))
        if isinstance(root, dict):
            self.recursive_split(root, curr_depth=1)
        
        # tree result-1. Every leaf node has data indices.
        self.estimator1 = root
        
        # tree result-2. Every leaf node has its output values.
        if isinstance(self.estimator1, dict):
            self.estimator2 = copy.deepcopy(self.estimator1)
            self.output_leaf(self.estimator2)           # tree result-2
        return self.estimator2

    # Estimate the target value of a test data point.
    def x_predict(self, p, x):
        if x[p['fid']] <= p['split_point']:
            if isinstance(p['left'], dict):  # recursion if not leaf.
                return self.x_predict(p['left'], x)
            else:                            # leaf
                return p['left']
        else:
            if isinstance(p['right'], dict): # not a leaf. recursion
                return self.x_predict(p['right'], x)
            else:                            # return the value in the leaf, if leaf.
                return p['right']
    
    # Estimate the target values for all x_test points.
    def predict(self, x_test):
        p = self.estimator2    # predictor
        
        if isinstance(p, dict):
            y_pred = [self.x_predict(p, x) for x in x_test]
            return np.array(y_pred)
        else:
            return np.array([self.base_score] * x_test.shape[0])

# Build XGBoost regression trees
class MyXGBRegressor:
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.3,
                 prune_gamma=0.0, reg_lambda=0.0, base_score=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = learning_rate        # learning rate
        self.prune_gamma = prune_gamma  # pruning constant
        self.reg_lambda = reg_lambda    # regularization constant
        self.base_score = base_score    # initial prediction        
        self.estimator1 = dict()        # tree result-1
        self.estimator2 = dict()        # tree result-2
        self.models = []
        self.loss = []
    
    # The same as GBM algorithm. In XGBoost, only the node 
    # splitting method changes.
    def fit(self, x, y):
        # step-1: Initialize model with a constant value.        
        Fm = self.base_score
        self.models = []
        self.loss = []
        for m in range(self.n_estimators):
            # step-2 (A): Compute so-called pseudo-residuals
            residual = y - Fm
            
            # step-2 (B): Fit a regression tree to the residual
            model = MyXGBRegressionTree(max_depth = self.max_depth,
                                        reg_lambda = self.reg_lambda,
                                        prune_gamma = self.prune_gamma)
            model.fit(x, residual)
            
            # step-2 (C): compute gamma (prediction)
            gamma = model.predict(x)
            
            # step-2 (D): Update the model
            Fm = Fm + self.eta * gamma
            
            # save tree models
            self.models.append(model)
            
            # Calculate the loss = mean squared error.
            self.loss.append(((y - Fm) ** 2).sum())
        return self.loss

    def predict(self, x_test):
        y_pred = np.zeros(shape=(x_test.shape[0],)) + self.base_score
        for model in self.models:
            y_pred += self.eta * model.predict(x_test)
        return y_pred


