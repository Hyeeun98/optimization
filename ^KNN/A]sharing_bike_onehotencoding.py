import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../csv/sharing_bike_train.csv')

# datetime 데이터를 날짜로 활용할 수 있도록 변환
df["year"] = pd.to_datetime(df["datetime"]).dt.year
df["month"] = pd.to_datetime(df["datetime"]).dt.month
df["day"] = pd.to_datetime(df["datetime"]).dt.day
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday
# 변환한 날짜데이터 카테고리화
df['year'] = df['year'].astype('category')
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')
df['hour'] = df['hour'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['season'] = df['season'].astype('category')

#변환한 datetime을 포함한 불필요 요소 제거
del df['datetime']
del df['casual']
del df['registered']
del df['count']

# one_hot_encoding
df = pd.get_dummies(df)

# train / test 데이터 분리
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

# train / test 데이터 중 weather 데이터를 수치로 라벨링
train_y = train['weather']
del train['weather']
train_x = train

test_y = test['weather']
del test['weather']
test_x = test

# 모델 생성 _비교할 인근 클러스터 설정
knn = KNeighborsClassifier(n_neighbors=2)
# knn을 train 데이터를 활용하여 학습
knn.fit(train_x,train_y)

# test 데이터를 활용하여 정확도 판단
score = knn.score(test_x,test_y)
print(score)