import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import random

data = []
for i in range(1000):
    data.append([random.randrange(1, 100), random.randrange(1, 100)])

#data = [[7, 1], [2, 1], [4, 2], [9, 4], [10, 5], [10, 6], [11, 5], [11, 6], [15, 3], [15, 2], [16, 4], [16, 1]]
df = pd.DataFrame(columns=['x', 'y'], data=data)
km = KMeans(n_clusters=5).fit(df[['x', 'y']])
df['cluster_id'] = km.labels_

sns.lmplot('x', 'y', data=df, hue='cluster_id', fit_reg=False)
plt.show()
