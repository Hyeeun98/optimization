import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics


df = pd.read_csv('../csv/titanic.csv') #'.'는 폴더 하나 의미 / ..일 경우 폴더 두개 건너간다는 뜻


del df['PassengerId'] #불필요한 값 제거
del df['Name']
del df['Ticket']
del df['Cabin']

df = pd.get_dummies(df, columns=['Sex','Embarked']) #문자 형식으로 된 값 인식가능한 형태 문자로 변환(one hot encoding)

df['Age'] = df['Age'].fillna(df['Age'].mean())
# age에 값이 없어서 실행이 안될 경우 사용
# fillna 로 넣을 값 지정_ 평균, 중간값 혹은 지우기


print(df.columns.tolist())

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

train_y = train['Survived']  #dummies때문에 label 형태가 달라지기 떄문에 인식못하는 문제 해결
del train['Survived']
train_x = train

test_y = test['Survived']
del test['Survived']
test_x = test

print(test_x.columns.to_list())

print(train.columns.tolist())

logistic = LogisticRegression(solver='newton-cg')
logistic.fit(train_x, train_y) # 바뀐 적용된 변수 입력
score = logistic.score(test_x, test_y)
print(score)

