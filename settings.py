#!/usr/bin/env python
# coding: utf-8

"""
定数の定義と必要なパッケージのインポート
先にこのモデルの学習とテストで利用する定数を定義しておきます。

future_numでは、価格が上がるか下がるかを予測する未来の10分足数です。ここでは10分足データの144足分のため、1日先の価格が上がるか下がるか、の予測となります。
feature_numは入力データの特徴量の数で、ボリューム、Open, High, Low, Closeの5項目を利用します。
batch_sizeはLSTMの学習時に一度に投入するデータポイント数です。
time_stepsは、LSTMが予測で利用する過去のデータポイントの数です。今回は過去の50個分のデータを見て、144個先のClose値が現在に比べて上がるのか下がるのかを予測するモデルとしています。
moving_average_numで500と指定しています。これは、LSTMに投入するデータは過去500足分の移動平均に対する現在の値の比率とするためです。
n_epochsはLSTMのトレーニングで何epoch数分実施するかです。
val_idx_from、test_idx_fromはそれぞれデータの何行目以降を評価用、テスト用として分割するかの位置です。
lstm_hidden_dim, target_dimはLSTMの隠れ層の出力サイズと最終出力サイズです。
"""

#何足先を予測するか
#future_num = 144
future_num = 96 # 翌日の同時間帯がどれだけ後か 15分足だと1440 / 15 = 96

#volume, open, high, low, closeの5項目
feature_num = 5 #特徴量

batch_size = 128 # LSTMに渡すレコード数のこと

# lstmのtimesteps
time_steps = 50

# 移動平均を取るCandle数
moving_average_num = 500
#moving_average_num = 300

n_epocs = 30
#n_epocs = 5 #学習する回数

#データをtrain, testに分割するIndex
#val_idx_from = 80000
#test_idx_from = 100000
val_idx_from  = 80000   # csvのこれより前のレコードを学習に使用する
test_idx_from = 100000  # csvのval_idx_fromより後で、これより前のレコードを評価に使用して、これより後のレコードをテスト用データとする

lstm_hidden_dim = 16 # LSTMの隠れ層の出力サイズ
target_dim = 1       # LSTMの隠れ層の最終出力サイズ
