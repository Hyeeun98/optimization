from sklearn.datasets import fetch_california_housing
import pandas as pd

california_data = fetch_california_housing()
california = pd.DataFrame(data=california_data.data, columns=california_data.feature_names)
print(california)