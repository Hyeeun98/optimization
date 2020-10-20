import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('../csv/seoul.csv')
sns.lmplot('lat', 'lon', data=df, hue='name', fit_reg=False)
label_count = len(df['name'].unique())
plt.show()

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

knn = KNeighborsClassifier(n_neighbors=label_count)  # 덩어리 수
knn.fit(train[['lat', 'lon']], train['name'])  # fit(train[['속성1','속성2']] , train['레이블'])
score = knn.score(test[['lat', 'lon']], test['name'])
print(score)


guess = pd.DataFrame(columns=['lat','lon'])
guess.loc[0]=[37.501690, 127.133596] # input data
print(knn.predict(guess))
