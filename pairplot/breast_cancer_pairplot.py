import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../csv/breast_cancer.csv')
sns.pairplot(data=df[
    ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
     "concavity_mean", "concave points_mean"]], hue='diagnosis')
plt.show()
