# [MXML-8-07] 8.iForest_outlier.py
# Outlier detection using Isolation Forest (iForest)
#
# This code was used in the machine learning online 
# course provided by 
# www.youtube.com/@meanxai
# www.github.com/meanxai/machine_learning
#
# A detailed description of this code can be found in
# https://youtu.be/JpZJoOTjMWU
# 
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Create training dataset
x, y = make_blobs(n_samples=600, n_features=2, 
                centers=[[0., 0.], [0.5, 0.5]], 
                cluster_std=0.2, center_box=(-1., 1.))

model = IsolationForest(n_estimators = 50, contamination=0.05)
model.fit(x)
outlier = model.predict(x)   # Normal = 1, Outlier = -1

# Extract outliers
i_outlier = np.where(outlier == -1)[0]
x_outlier = x[i_outlier, :]

# Visualize normal data points and outliers by color.
plt.figure(figsize=(7, 7))
color = [['blue', 'red'][i] for i in y]
color_out = [['blue', 'red'][i] for i in y[i_outlier]]
plt.scatter(x[:, 0], x[:, 1], s=30, c=color, alpha=0.5)
plt.scatter(x_outlier[:, 0], x_outlier[:, 1], s=400, c='black', alpha=0.5)  # outlier scatter
plt.scatter(x_outlier[:, 0], x_outlier[:, 1], s=200, c='white')
plt.scatter(x_outlier[:, 0], x_outlier[:, 1], s=30, c=color_out)
plt.show()

# Check out the distribution of Anomaly score
score = abs(model.score_samples(x))
score[i_outlier].min()
plt.hist(score, bins = 50)
plt.title('distribution of anomaly score')
plt.xlabel('anomaly score')
plt.ylabel('frequency')
plt.axvline(x=score[i_outlier].min(), c='red')
plt.show()


