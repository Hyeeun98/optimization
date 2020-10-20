import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('../csv/fruit_data_with_colors.csv')
sns.lmplot('width', 'height', data=df, hue='fruit_name', fit_reg=False)
label_count = len(df['fruit_name'].unique())
plt.show()

train = df.sample(frac=0.00001, random_state=200)
test = df.drop(train.index)

knn = KNeighborsClassifier(n_neighbors=label_count)  # 덩어리 수
knn.fit(train[[ 'width','height']], train['fruit_name'])  # fit(train[['속성1','속성2']] , train['레이블'])
score = knn.score(test[['width','height']], test['fruit_name'])
print(score)    


guess = pd.DataFrame(columns=['width','height'])
guess.loc[0]=[6,4] # input data
print(knn.predict(guess))
