from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../csv/sharing_bike_train.csv')

df["year"] = pd.to_datetime(df["datetime"]).dt.year
df["month"] = pd.to_datetime(df["datetime"]).dt.month
df["day"] = pd.to_datetime(df["datetime"]).dt.day
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday

df["year"] = df["year"].astype("category")
df["month"] = df["month"].astype("category")
df["day"] = df["day"].astype("category")
df["hour"] = df["hour"].astype("category")
df["weekday"] = df["weekday"].astype("category")

del df["datetime"]




train = df.sample(frac=0.9, random_state=200)
test = df.drop(train.index)
sns.pairplot(data=test[['weather','temp', 'atemp', 'humidity', 'windspeed']],hue='weather')

#sns.pairplot(data=df[['weather','temp', 'atemp', 'humidity', 'windspeed']],hue='weather')



plt.show()
