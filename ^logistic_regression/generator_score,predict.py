import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv('../csv/generator.csv')
sns.lmplot('RPM', 'VIBRATION', data=df, hue='STATUS', fit_reg=False)
plt.show()

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

logistic = LogisticRegression()
logistic.fit(train[['RPM', 'VIBRATION']], train['STATUS'])
score = logistic.score(test[['RPM', 'VIBRATION']], test['STATUS'])
print(score)

guess = pd.DataFrame(columns=['RPM','VIBRATION'])
guess.loc[0]=[600,600]
print(logistic.predict(guess))
