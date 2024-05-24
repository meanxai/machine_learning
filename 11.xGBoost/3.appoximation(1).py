# [MXML-11-07] 3.approximation(1).py
# 논문 [1] Tianqi Chen et, al., 2016, XGBoost: A Scalable Tree Boosting System
# 3. SPLIT FINDING ALGORITHMS
# 3.2 Approximate Algorithm
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/AQOPXlxXF_0
# 
import numpy as np
from MyXGBoostRegressor import MyXGBRegressor
import time

# Create training data
def nonlinear_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.random()
       y = 2.0 * np.sin(2.0 * np.pi * x) + \
           np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)
x, y = nonlinear_data(n=50000, s=0.5)

# 1. Exact Greedy Algorithm (EGA)
# -------------------------------
start_time = time.time()
my_model = MyXGBRegressor(n_estimators = 1,
                          max_depth = 1,
                          base_score = y.mean())

my_model.fit(x, y)
e = my_model.models[0].estimator2

print('\nExact greedy algorithm:')
print('split point =', np.round(e['split_point'], 3))
print('gain =', np.round(e['gain'], 3))
print('running time = {:.2f} seconds'.format(time.time() - start_time))

# 2.Approximate Algorithm (AA).
# -------------------------------
from multiprocessing.pool import Pool
def find_split_point(x, y):
    # MyXGBRegressor is a class implemented with EGA. 
    # To implement this properly, you need to implement the 
    # Approximate Algorithm inside the MyXGBRegressor.
    my_model = MyXGBRegressor(n_estimators = 1,
                              max_depth = 1,        # root node만 확인함.
                              base_score = y.mean())

    my_model.fit(x, y)
    e = my_model.models[0].estimator2
    return [e['split_point'], e['gain']]
    
# Divide the data into five parts and allocate 20% of the data to
# each part.
c_point = np.percentile(x, [20, 40, 60, 80, 100])

# maps the data into buckets split by c_point
l_bound = -np.inf
x_block, y_block = [], []
for p in c_point:
    idx = np.where(np.logical_and(x > l_bound, x <= p))[0]
    x_block.append(x[idx])
    y_block.append(y[idx])
    l_bound = p

start_time = time.time()
mp = Pool(5)
args = [[ax, ay] for ax, ay in zip(x_block, y_block)]
ret = mp.starmap_async(find_split_point, args)
mp.close()
mp.join()

print('\nApproximate Algorithm:')
print('split_points =', np.array(ret.get())[:, 0].round(3))
print('gain =', np.array(ret.get())[:, 1].round(2))
print('running time = {:.2f} seconds'.format(time.time() - start_time))
print('number of data in blocks =', [len(a) for a in x_block])