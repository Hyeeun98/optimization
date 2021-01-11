import csv

a4csv = open('../csv/a4.csv', 'r', encoding='utf-8')
a4 = csv.reader(a4csv)
a4_array = []
for i in a4: a4_array.append(i)

purchase = a4_array[0][1:]
sales = a4_array[1][1:]

for i in range(len(purchase)): purchase[i] = int(purchase[i])
for i in range(len(sales)): sales[i] = int(sales[i])

sales_4w_avr = sum(sales[len(sales) - 4: len(sales)]) / 4

maxSales = max(sales)
AVRLEAD = 1
MAXLEAD = 2

safeInven = (maxSales * MAXLEAD) - (sales_4w_avr)
safeInvenCycle = safeInven / sales_4w_avr

print(safeInvenCycle)







## all new
stock_history = []
max_sell = 0
max_lead = 2
average_lead = 1
safe_stock = 0
safe_period = 0

buy_arr = purchase
sell_arr = sales


for i in range(len(buy_arr)):
    buy = buy_arr[i]
    sell = sell_arr[i]

    # 반복시 초기화 되는 값들
    remain = 0
    past_stocks = stock_history[-3:]
    item_count = len(past_stocks) + 1
    sell_average = 0
    last_stock = 0
    total_sell = sell

    #
    for past in past_stocks:
        total_sell = total_sell + past['sell']
        last_stock = past['stock']

    sell_average = total_sell // item_count
    if max_sell < sell: max_sell = sell

    # 재고량, 안전 재고 정보 구하기
    if len(stock_history) == 0:
        remain = buy - sell
        last_stock = remain
    else:
        remain = buy - sell + last_stock

    stock_period = remain / sell_average
    safe_stock = round(max_sell * max_lead - sell_average * average_lead)
    safe_period = round(safe_stock / sell_average, 1)

    buy_recommend = round(safe_period * sell_average - last_stock + sell)
    if buy_recommend < 0: buy_recommend = 0

    stock = {
        'buy': buy,
        'sell': sell,
        'stock' : remain,
        'stock_period': stock_period,
    }

    stock_history.append(stock)
    print("=========%d========="%i)
    print("구매량 : %d" % (buy,))
    print("판매 : %d" % (sell,))
    print("평균 판매 : %d" % (sell_average,))
    print("재고 : %d" % (remain,))
    print("재고주기 : %d" % (stock_period,))
    print("안전 재고 : %d" % (safe_stock,))
    print("안전 재고 주기 : %d" % (safe_period,))
    print("추천 구매량 : %d" % (buy_recommend,))
    print("\n\n\n\n")
