#!/usr/bin/env python
# coding: utf-8

import os
import requests
import datetime
import pandas as pd

def klines(symbol="BTCUSDT", interval="15m", startTime=0, endTime=0, limit=500):
    # APIで価格データを取得
    params = {"symbol" : symbol, "interval" : interval, "limit" : limit }
    if startTime != 0:
        params["startTime"] = startTime
    if endTime != 0:
        params["endTime"] = endTime
    response = requests.get("https://api.binance.com/api/v3/klines",params)
    data = []
    for res in response.json():
        s = res[0] / 1000.0
        data.append( [
                datetime.datetime.fromtimestamp(s),
                res[5],
                res[1],
                res[2],
                res[3],
                res[4]
            ])
    return data

if not os.path.exists("./data"):
    os.mkdir("./data")

try:
    _endTime = 0
    while True:
        test = klines(endTime=_endTime)
        df = pd.DataFrame(test[1:])
        result = test[1:]
        #file_name = result[0][0].strftime('%Y%m%d%H%M%S') + "-" + result[len(result) - 1][0].strftime('%Y%m%d%H%M%S') + ".csv"
        file_name = result[0][0].strftime('%Y%m%d%H%M') + "-" + result[len(result) - 1][0].strftime('%Y%m%d%H%M') + ".csv"
        df = pd.DataFrame(result)
        df.columns = ['Datetime', 'Volume', 'Open', 'High', 'Low', 'Close']
        df = df.set_index('Datetime')
        df.to_csv('./data/' + file_name, header=False)
        _endTime = int(test[0][0].timestamp() * 1000)
except Exception as e:
    print(e.args)
    print('cat *.csv > test.txt')
    print('cat test.txt | sort > data.csv')
