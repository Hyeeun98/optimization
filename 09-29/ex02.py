import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('../csv/diabetes.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'result'])
sns.pairplot(data=df[
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'result']], hue='result')
plt.show()


train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

logistic = LogisticRegression()
logistic.fit(train[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']], train['result'])
score = logistic.score(test[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']], test['result'])
print(score)
