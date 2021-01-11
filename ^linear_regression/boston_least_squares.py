from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model import LinearRegression

# 목적: 주택의 가격에 영향을 미치는 요소를 분석
boston_data = load_boston()
#dataframe으로 정제
boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
boston['target'] = boston_data.target

train = boston.sample(frac=0.8, random_state=200)
test = boston.drop(train.index)

mlr = LinearRegression()
mlr.fit(train[['PTRATIO', 'INDUS', 'NOX', 'B', 'CHAS', 'RAD', 'TAX', 'ZN', 'DIS', 'CRIM','RM', 'LSTAT', 'AGE']],
        train['target'])

# coef(coefficient,계수)
# intercept
# coef중에서 intercept, weight 값을 찾고, 그 값을 모델에 대입
print(mlr.intercept_)  #절편 출력
print(mlr.coef_)  #계수 출력

#####california_housing 항목 값으로 바꾸기
sum_difference = 0
for i, row in test.iterrows():
    estimate = row['PTRATIO'] * mlr.coef_[0] + row['INDUS'] * mlr.coef_[1] + row['NOX'] * mlr.coef_[2] + \
               row['B'] * mlr.coef_[3] + row['CHAS'] * mlr.coef_[4] + row['RAD'] * mlr.coef_[5] + \
               row['TAX'] * mlr.coef_[6] + row['ZN'] * mlr.coef_[7] + row['DIS'] * mlr.coef_[8] + \
               row['CRIM'] * mlr.coef_[9] + row['RM'] * mlr.coef_[10] + row['LSTAT'] * mlr.coef_[11] + \
               row['AGE'] * mlr.coef_[12] + mlr.intercept_
    sum_difference += abs(estimate - row['target'])
print(sum_difference)
