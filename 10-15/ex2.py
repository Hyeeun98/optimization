import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics

# ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']


df = pd.read_csv('../csv/bank_marketing_simple.csv', sep=';')
df2 = pd.read_csv('../csv/bank_marketing_full.csv', sep=';')


"""
del df['job']
del df['marital']
del df['education']
del df['contact']
del df['month']
del df['poutcome']
del df['y']
"""


df = pd.get_dummies(df,columns=['job','marital','education','default', 'housing', 'loan','contact','day','month','poutcome'])

df2 = pd.get_dummies(df2,columns=['job','marital','education','default', 'housing', 'loan','contact','day','month','poutcome'])
print(df.columns.tolist())

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

train_y = train['y']
del train['y']
train_x = train

test_y = test['y']
del test['y']
test_x = test


logistic = LogisticRegression(solver='newton-cg')
logistic.fit(train_x,train_y)
score = logistic.score(test_x,test_y)
print(score)