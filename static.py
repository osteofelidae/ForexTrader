# FILE: Static algorithm for either baseline or just do it


# DEPENDENCIES
import data as d
import numpy as np
import analysis as a


# MAIN
data_b = d.load(path="datasets/AUD_USD/5.csv")
data = d.load(path="datasets/AUD_USD/5.csv")  # Load data

# TODO try normalizing

bids_all = data[:, 0]
asks_all = data[:, 1]

holding = False
holding_price = 0
prev_spread = abs(data[0][0]-data[0][1])
prev_bid = data[0][0]
prev_ask = data[0][1]
count = 0

#margin_div = 100
margin_div = 100

rsi_threshold = 30

lower_band_ask = min(data_b[:, 1]) + (max(data_b[:, 1]) - min(data_b[:, 1]))/margin_div
upper_band_bid = max(data_b[:, 0]) - (max(data_b[:, 0]) - min(data_b[:, 0]))/margin_div

for index1 in range(len(data)-1):
    index = index1 + 1
    point = data[index]

    scoped = d.scope(data=data, index=index, length=5)
    bids = scoped[:, 0]
    asks = scoped[:, 1]

    bid = point[0]
    ask = point[1]

    bid_avg = np.average(bids)
    ask_avg = np.average(asks)

    spread = abs(ask - bid)

    up = 1e-8  # Total profit: set to small number epsilon
    down = 1e-8  # Total loss
    for ind in range(len(scoped) - 1):  # Iterate over length of spot array
        if scoped[ind][0] < scoped[ind + 1][0]:  # If profit
            up += scoped[ind + 1][0] - scoped[ind][0]  # Add to profit total
        elif scoped[ind][0] > scoped[ind + 1][0]:  # If loss
            down += scoped[ind][0] - scoped[ind + 1][0]  # Add to loss total
    rs = up / down  # Relative strength
    rsi = (100 - (100 / (1 + rs)))  # Relative strength index for index 0

    lower_ask = min(scoped[:, 1]) + (max(scoped[:, 1]) - min(scoped[:, 1])) / margin_div
    upper_bid = max(scoped[:, 0]) - (max(scoped[:, 0]) - min(scoped[:, 0])) / margin_div

    #buy_signal = (ask <= min(bids)) and (rsi < 50)
    #sell_signal = (bid > holding_price * 1.002) and (rsi > 50)

    buy_signal = (ask < lower_band_ask)# and (rsi <= 100-rsi_threshold)
    sell_signal = (bid > upper_band_bid)# and (rsi >= rsi_threshold)

    #buy_signal = (ask < lower_ask)# and (rsi <= 100 - rsi_threshold)
    #sell_signal = (bid > upper_bid)# and (rsi >= rsi_threshold)

    if buy_signal and not holding:

        holding = True
        holding_price = ask
        a.t_buy(point=point, units=100)
        count = 0


    elif sell_signal and holding:

        holding = False
        a.t_sell_all(point=point)

    elif holding:
        count += 1

    prev_spread = spread
    prev_ask = ask
    prev_bid = bid

    print(a.t_balance, holding)

a.t_sell_all(point=data[-1])

print(a.t_balance)
print("=====", max(data[:, 0]), min(data[:, 1]))