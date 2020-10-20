from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../csv/sharing_bike_train.csv')

del df["datetime"]
del df['causal']
del df['registered']
"""
One Hot Encoding

a,b,c
=====
1,0,0
0,1,0
0,0,1
"""

df = pd.get_dummies(df) #one hot encoding


