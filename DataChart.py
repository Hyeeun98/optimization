import csv
from matplotlib import pyplot as plt

sale_data = []
with open('./csv/sample_data.csv', encoding='utf-8-sig') as data:
    reader = csv.DictReader(data)
    for row in reader:
        price = int(row['price'])
        sale_dty = int(row['sale_qty'])

        sale_data.append({
            'price': price,
            'qty': sale_dty,
        })
        plt.scatter(price, sale_dty)

    # price -> x // sale_dty -> y
    # 기울기
    m = (sale_data[len(sale_data) - 1]['price'] - sale_data[0]['price']) / (
            sale_data[len(sale_data) - 1]['qty'] - sale_data[0]['qty'])
    #  plt.plot(x, [y] * len(x))
    # plt.show()
    minimum_diff = -1
    for divider in range(-100, 101):
        if divider == 0: continue
        for dividend in range(1, 101):
            weight = dividend / ( divider * 10000)  # 가중치

            sum_diff = 0
            for data in sale_data:
                estimate = data.get('price') * weight
                diff = estimate - data.get('qty')
                sum_diff += diff  # 예상값과 실제 값의 차이를 더해주는 것
            if minimum_diff == -1 or sum_diff < minimum_diff:
                # 최저 차이를 보이는 가중치인 경우
                optimal_weight = weight
                minimum_diff = sum_diff
    # 차트 데이터 만들어야 함
    y_axis = []
    x_axis = []
    for price in range(10000, 100000, 1000):
        estimate_qty = optimal_weight * price / 1000
        x_axis.append(price)
        y_axis.append(estimate_qty)
    plt.plot(x_axis, y_axis)
    plt.show()
