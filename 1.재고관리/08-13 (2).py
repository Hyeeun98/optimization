import csv
import matplotlib.pyplot as plt

sale_data = []
optimal_weight = 0
optimal_xaxis = []  # 가격
optimal_yaxis = []  # 예상 판매랑

sum_x = 0
sum_y = 0
with open('./csv/sample_data.csv', encoding='utf-8-sig') as data:
    reader = csv.DictReader(data)
    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])

        sale_data.append({
            'price': price,
            'qty': sale_qty,
        })
        sum_x += price
        sum_y += sale_qty
        plt.scatter(price, sale_qty)

    # 전체 평균
    avg_x = sum_x / len(sale_data)
    avg_y = sum_y / len(sale_data)

    divider = 0
    diviend = 0
    
    # https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F995D9F505C66F17105
    # 최소 제곱법 공식 참조
    for data in sale_data:
        divider += (data.get('price') - avg_x) ** 2  #
        diviend += (data.get('price') - avg_x) * (data.get('qty') - avg_y)

    optimal_weight = diviend / divider  # 가중치
    b = avg_y - (avg_x * optimal_weight) 


    for price in range(10000, 100000, 1000):
        estimate_qty = optimal_weight * price + b
        optimal_xaxis.append(price)
        optimal_yaxis.append(estimate_qty)
    plt.plot(optimal_xaxis, optimal_yaxis)
    plt.show()


