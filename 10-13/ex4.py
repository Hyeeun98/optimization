# R2 score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('../csv/sharing_bike_train.csv')

df['year'] = pd.to_datetime(df['datetime']).dt.year
df['month'] = pd.to_datetime(df['datetime']).dt.month
df['day'] = pd.to_datetime(df['datetime']).dt.day
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
df['weekday'] = pd.to_datetime(df['datetime']).dt.weekday

del df['datetime']
del df['casual']
del df['registered']

df['year'] = df['year'].astype('category')
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')
df['hour'] = df['hour'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['season'] = df['season'].astype('category')
df['weather'] = df['weather'].astype('category')

df = pd.get_dummies(df)

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

train_y = train['count']
del train['count']
train_x = train

test_y = test['count']
del test['count']
test_x = test

mlr = LinearRegression()
mlr.fit(train_x, train_y)

prediction = mlr.predict(train_x)
score = metrics.r2_score(train_y, prediction)
print(score)

prediction = mlr.predict(test_x)
score = metrics.r2_score(test_y, prediction)
print(score)
