import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../csv/sharing_bike_train.csv')

df["year"] = pd.to_datetime(df["datetime"]).dt.year
df["month"] = pd.to_datetime(df["datetime"]).dt.month
df["day"] = pd.to_datetime(df["datetime"]).dt.day
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday

df['year'] = df['year'].astype('category')
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')
df['hour'] = df['hour'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['season'] = df['season'].astype('category')


del df['datetime']
del df['casual']
del df['registered']
del df['count']

df = pd.get_dummies(df)
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

train_y = train['weather']
del train['weather']
train_x = train

test_y = test['weather']
del test['weather']
test_x = test

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train_x,train_y)
score = knn.score(test_x,test_y)
print(score)