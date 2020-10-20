import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('../csv/COVID-19-geographic-disbtribution-worldwide-2020-09-29.csv')
sns.pairplot(data=df[
    ['cases','deaths','Cumulative_14_days_cases_100000']], hue='deaths')
plt.show()

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)