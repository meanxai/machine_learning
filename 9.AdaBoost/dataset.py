import numpy as np
from sklearn.utils import shuffle

# regression test 용 비선형 1D 데이터를 생성한다.
def noisy_sine_data(n, s):
   rtn_x, rtn_y = [], []
   for i in range(n):
       x = np.random.random()
       y = 2.0 * np.sin(2.0 * np.pi * x) + np.random.normal(0.0, s) + 3.0
       rtn_x.append(x)
       rtn_y.append(y)
       
   return np.array(rtn_x).reshape(-1,1), np.array(rtn_y)

# classification 시험용 데이터를 생성한다.
def blob_data(n, locs, scales):
    for i, (loc, scale) in enumerate(zip(locs, scales)): 
        x1 = np.random.normal(loc, scale, (n[i], 2))
        y1 = np.ones(n[i]).astype('int') * i
        
        if i == 0:
            x = x1
            y = y1
        else:
            x = np.vstack([x, x1])
            y = np.hstack([y, y1])
    
    return shuffle(x, y)
