import csv
import numpy as np
import matplotlib.pyplot as plt

with open('./csv/sample_data.csv', encoding='utf-8-sig') as data:
    reader = csv.DictReader(data)
    prices = []
    quantities = []
    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])
        prices.append(price)
        quantities.append(sale_qty)
        plt.scatter(price, sale_qty)
    x = np.array(prices)
    y = np.array(quantities)
    fit = np.polyfit(x, y, 2)  # ㅎ 다차 방정식에 대한 최적값을 회귀분석 해줌
    # polyfit(x차수 , y차수, 계수)
    optimal_xaxis = []
    optimal_yaxis = []
    print(fit)
    for price in range(10000, 100000, 1000):
        optimal_xaxis.append(price)
        optimal_yaxis.append(fit[0] * (price ** 2) + fit[1] * price + fit[2])
    plt.plot(optimal_xaxis, optimal_yaxis)
    plt.show()
