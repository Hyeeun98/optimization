import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

data = [[7, 1], [2, 1], [4, 2], [9, 4], [10, 5], [10, 6], [11, 5], [11, 6], [15, 3], [15, 2], [16, 4], [16, 1]]

#df = pd.DataFrame(columns=['x', 'y'], data=data)
df = pd.read_csv('./csv/cluster_sample_1.csv')
distortions = []

for cluster in range(1, 10):
    km = KMeans(n_clusters=cluster).fit(df[['x', 'y']])
    #중심점과 모든 좌표들간의 거리 (N:M)
    distance = cdist(df[['x', 'y']], km.cluster_centers_, 'euclidean')
    #
    min_distance = np.min(distance, axis = 1)
    sum_distance = np.sum(min_distance)

    distortions.append(sum_distance/df[['x','y']].shape[0])


km = KMeans(n_clusters=15).fit(df[['x', 'y']])
df['cluster_id'] = km.labels_

sns.lmplot('x', 'y', data=df, hue='cluster_id', fit_reg=False)
plt.show()

#plt.scatter(df[['x']],df[['y']])
#plt.plot(range(1,10), distortions)

plt.show()
