from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../csv/example_retail_sales.csv')

m = Prophet()
forecast = m.fit(df)
future = m.make_future_dataframe(periods=200)

future = m.predict(future)

plt.plot(future['ds'][250:], future['trend'][250:])
plt.plot(future['ds'][250:], future['yhat'][250:])
plt.plot(future['ds'][250:], future['yhat_lower'][250:])
plt.plot(future['ds'][250:], future['yhat_upper'][250:])

x_values = []
y_values = []

for i, row in df[250:].iterrows():
    y_values.append(future['ds'][i])
    x_values.append(row['y'])

plt.plot(y_values, x_values)

plt.show()

