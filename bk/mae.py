#!/usr/bin/env python
# coding: utf-8

import urllib.request
import sys
import datetime
import calendar
import os
import gzip

def downloadcsv(year, month):
    lastday = calendar.monthrange(year, month)[1]
    _month = f'{month:02}'
    _folder = str(year) + _month
    if not os.path.exists("./data/" + _folder):
        os.mkdir("./data/" + _folder)
    for x in range(1, lastday + 1):
        _day = f'{x:02}'
        file_name = str(year) + _month + _day + "_BTC.csv.gz"
        url = "https://api.coin.z.com/data/trades/BTC/" + str(year) + "/" + _month + "/" + file_name
        _full_path = "./data/" + _folder + "/" + file_name
        urllib.request.urlretrieve(url, _full_path)
    
downloadcsv(2018,9)
with gzip.open("./data/201809/20180905_BTC.csv.gz", mode='rt') as fp:
    data123 = fp.read()
print(data123.split('\n')[2])
