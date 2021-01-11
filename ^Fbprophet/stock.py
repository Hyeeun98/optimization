# 변화 빈도가 높은 데이터 # 실제로 예측가능한지 파악하는 것이 목표
# volume : 거래량 / adj close : 마감시 평균 가격

from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../csv/stock_sample.csv')

# 불필요한 항목 제거
del df['Open']
del df['High']
del df['Low']
del df['Volume']
del df['Adj Close']

# Date를 ds로 이름을 바꾼다
df = df.rename(columns = {
    'Date' : 'ds',
    'Close' : 'y',
})

# daily 데이터 (daily_seasonality)
m = Prophet(daily_seasonality=True)
forecast = m.fit(df)
future = m.make_future_dataframe(periods=10)

future = m.predict(future)

# 실제 데이터 출력
y_values = []
x_values = []
for i, row in df[190:].iterrows():
    y_values.append(future['ds'][i])
    x_values.append(row['y'])

# 그래프 출력
# 하한선을 기준으로 보면, 대부분 범위안에 들어오기 때문에 수익이 생기겠다.라고 판단 가능
plt.plot(y_values, x_values)
plt.plot(future['ds'][190:], future['trend'][190:])
plt.plot(future['ds'][190:], future['yhat'][190:])
plt.plot(future['ds'][190:], future['yhat_lower'][190:])
plt.plot(future['ds'][190:], future['yhat_upper'][190:])

plt.show()

