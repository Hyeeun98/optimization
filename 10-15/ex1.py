import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics

# ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']


df = pd.read_csv('../csv/bank_marketing_simple.csv', sep=';')


del df['previous']
del df['pdays']
del df['contact']
del df['duration']
del df['month']
del df['day']
del df['campaign']

df['job'] = df['job'].astype('category')
df['marital'] = df['marital'].astype('category')
df['education'] = df['education'].astype('category')
df['poutcome'] = df['poutcome'].astype('category')

"""
del df['job']
del df['marital']
del df['education']
del df['contact']
del df['month']
del df['poutcome']
del df['y']
"""


df = pd.get_dummies(df,columns=['job','marital','education','poutcome','default', 'housing', 'loan'])
print(df.columns.tolist())

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

print(train.columns.tolist())

logistic = LogisticRegression()
logistic.fit(train[['age', 'default', 'balance', 'housing', 'loan', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single', 'education_primary', 'education_secondary', 'education_tertiary', 'education_unknown', 'poutcome_failure', 'poutcome_other', 'poutcome_success', 'poutcome_unknown']], train['y'])
score = logistic.score(test[['age', 'default', 'balance', 'housing', 'loan', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single', 'education_primary', 'education_secondary', 'education_tertiary', 'education_unknown', 'poutcome_failure', 'poutcome_other', 'poutcome_success', 'poutcome_unknown']], test['y'])
print(score)


