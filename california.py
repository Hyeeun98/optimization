import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

california_data = fetch_california_housing()
california = pd.DataFrame(data=california_data.data, columns=california_data.feature_names)
california['target'] = california_data.target

train = california.sample(frac=0.8, random_state=200)
test = california.drop(train.index)

result = sm.ols(formula='target ~ CRIM + INDUS + ZN + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',data=train).fit()
print(result.summary())


sum_difference = 0
for i, row in test.iterrows():
    params = result.params
    r_estimate = row['PTRATIO'] * params['PTRATIO'] + row['INDUS'] * params['INDUS'] + row['NOX'] * params['NOX'] + \
                 row['B'] * params['B'] + row['CHAS'] * params['CHAS'] + row['RAD'] * params['RAD'] +\
                 row['TAX'] * params['TAX'] + row['ZN'] * params['ZN'] + row['DIS'] * params['DIS'] +\
                 row['CRIM'] * params['CRIM'] + row['RM'] * params['RM'] + row['LSTAT'] * params['LSTAT'] +\
                 params['Intercept']
    sum_difference += abs(r_estimate - row['target'])

print(sum_difference)
#scatter_matrix(california)
#plt.show()