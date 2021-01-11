from datetime import datetime
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../csv/example_retail_sales.csv')

last_date_str = df.iloc[len(df)-1]['ds'] # 마지막 날짜를 뽑아서 문자열 형태인 것을 날짜처리한것
last_date = datetime.strptime(last_date_str, '%Y-%m-%d') # 컴퓨터가 처리할 수 있는 형태로_ strptime

#기준점_
pivot = 290

while True:
    m = Prophet(seasonality_mode = 'multiplicative')
    # 반복할때마다 새로 예측 #prophet은 한번만 사용가능
    # multiplicative :  일관적이지 않은 모드를 활용할 것
    forecast = m.fit(df) #fit_ 학습하라는 것.
    future = m.make_future_dataframe(periods=20, freq = 'MS') #MS : month

    future = m.predict(future) #future에 예측값 집어넣는 것

    plt.plot(future['ds'][pivot:], future['trend'][pivot:])  # pivot을 넣어야 출력되는 그래프 시작점이 변경됨
    plt.plot(future['ds'][pivot:], future['yhat'][pivot:])
    plt.plot(future['ds'][pivot:], future['yhat_lower'][pivot:])
    plt.plot(future['ds'][pivot:], future['yhat_upper'][pivot:])

    x_values = []
    y_values = []

    for i, row in df[pivot:].iterrows():
        y_values.append(future['ds'][i])
        x_values.append(row['y'])

    plt.plot(y_values, x_values)

    plt.show()

    next_value = int(input('Input next value :'))
    next_date = last_date + relativedelta(months = 1) #relativedelata : 날짜 계산_month에 1 더함.
    next_date_str = next_date.strftime('%Y-%m-%d') #y_1998 중 98 #Y_1998 중 1998
    last_date = next_date
    new_row = pd.DataFrame ({
        'ds' : last_date_str,
        'y' : next_value}, index = [0])
    print(new_row)
    df = df.append(new_row, ignore_index = True)  #새로운 날짜에, 추가로 입력한 값 추가
    print(next_date_str)
    pivot = pivot+1
    
