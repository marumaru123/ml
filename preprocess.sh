#!/bin/bash
cd data
cat *.csv | sort > data.csv
sed -i '1s/^/Datetime,Volume,Open,High,Low,Close\n/' data.csv
head data.csv
rm -fr 20*.csv
