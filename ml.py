#!/usr/bin/env python
# coding: utf-8

# LSTM機械学習
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import os
from lstmclassifier import LSTMClassifier
import settings

# deviceはGPUの利用可否に応じでcudaまたはcpuがセットされます。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

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
#future_num = 96 # 翌日の同時間帯がどれだけ後か 15分足だと1440 / 15 = 96

#volume, open, high, low, closeの5項目
#feature_num = 5 #特徴量

#batch_size = 128 # LSTMに渡すレコード数のこと

# lstmのtimesteps
#time_steps = 50

# 移動平均を取るCandle数
#moving_average_num = 500
#moving_average_num = 300

#n_epocs = 30
#n_epocs = 30 #学習する回数

#データをtrain, testに分割するIndex
#val_idx_from = 80000
#test_idx_from = 100000
#val_idx_from  = 80000   # csvのこれより前のレコードを学習に使用する
#test_idx_from = 100000  # csvのval_idx_fromより後で、これより前のレコードを評価に使用して、これより後のレコードをテスト用データとする

#lstm_hidden_dim = 16 # LSTMの隠れ層の出力サイズ
#target_dim = 1       # LSTMの隠れ層の最終出力サイズ

"""
データ準備
LSTMで学習できるようにデータを準備していきます。

Oanda APIで取得したCSVデータを読み込みます。
教師データとして、144足先のClose値と現在のClose値を比較し、上がって入れば1、下がっていれば0をセットします。
数量や価格はそのまま利用するのではなく、直近500足データの移動平均に対する率とします。
約3.5日分の移動平均に対して何%上下しているかを予測のためのインプットとします。
データを分割し、PyTorchで利用できるようにtorchのtensorに変換しておきます。
"""

# 1. CSVファイルの読み込み
df = pd.read_csv('./data/data.csv', index_col='Datetime')

# 2. 教師データの作成
future_price = df.iloc[settings.future_num:]['Close'].values #dfのうちfuture_num個目以降のCloseをfuture_priceとする
curr_price = df.iloc[:-settings.future_num]['Close'].values  #dfのうち後ろからfuture_num個目のCloseをcurr_priceとする
y_data_tmp = future_price - curr_price              # future_price - curr_price
y_data = np.zeros_like(y_data_tmp)                  # 教師データをゼロで初期化
y_data[y_data_tmp > 0] = 1                          # future_price - curr_priceがゼロより大きい場合1とする。
y_data = y_data[settings.moving_average_num:]                # 移動平均を算出する行以降を最終的な教師データとする
# 3. 価格の正規化
cols = df.columns
for col in cols:
    #df['Roll_' + col] = df[col].rolling(window=500, min_periods=500).mean() # 移動平均を算出してRoll_[カラム名]というカラムを作成して設定する
    df['Roll_' + col] = df[col].rolling(window=settings.moving_average_num, min_periods=settings.moving_average_num).mean()
    df[col] = df[col] / df['Roll_' + col] - 1                           # カラム / 移動平均 - 1

#最初の500足分は移動平均データがないため除く。後半の144足分は予測データがないため除く
X_data = df.iloc[settings.moving_average_num:-settings.future_num][cols].values

# 4. データの分割、TorchのTensorに変換
#学習用データ(X_dataとy_dataのインデックスがval_idx_from件数までを学習用データとする)
X_train = torch.tensor(X_data[:settings.val_idx_from], dtype=torch.float, device=device)
y_train = torch.tensor(y_data[:settings.val_idx_from], dtype=torch.float, device=device)
#評価用データ(X_dataとy_dataのインデックスがval_idx_from件～test_idx_from件までを評価用データとする)
X_val   = torch.tensor(X_data[settings.val_idx_from:settings.test_idx_from], dtype=torch.float, device=device)
y_val   = y_data[settings.val_idx_from:settings.test_idx_from]
#テスト用データ(X_dataとy_dataのインデックスがtest_idx_from件以降をテスト用データとする)
X_test  = torch.tensor(X_data[settings.test_idx_from:], dtype=torch.float, device=device)
y_test  = y_data[settings.test_idx_from:]

"""
次に一つヘルパーファンクションを定義しておきます。
このファンクションは重要で、データポイントのindexのバッチ数分の配列を受けたら、
その各index毎に過去50個分の過去データを2つめの次元に追加してそれを一つの固まりとしてLSTMに投入できるようにします。

バッチ毎の処理数が128、特徴量の数(ボリューム、Open, High, Low, Close）が5のため、
このファンクションの入力データ（X_data）の次元は（128, 5)となります。

この各データポイントに対して、過去50個分（time_steps数）のデータを合成してfeatsとして返します。
そのため、戻り値の次元は(128, 50, 5）となります。2次元目に合成されたデータが過去50個分の時系列データとなります。
"""
def prep_feature_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去のN足分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats
"""
LSTM学習の実施

ここまで準備が整ったら、実際に学習を実施してみましょう。
LSTMのインスタンスを生成し、損失関数と最適化関数を設定します。

loss functionは二値分類（上がるか下がるか）なので、
素直にbinary classification entropy loss（BCELoss）を利用、optmizerはAdamを利用します。
"""
# Prepare for training
# モデルと損失関数とオプティマイザの設定
model = LSTMClassifier(settings.feature_num, settings.lstm_hidden_dim, settings.target_dim).to(device)
loss_function = nn.BCELoss()
optimizer= optim.Adam(model.parameters(), lr=1e-4)

"""
学習を実行していきます。

1.時系列処理とはいえ、全件を1件づつ回していくと時間がかかるので、ミニバッチを作るためにIndexをランダムに入れ替えます。
2.ランダムに入れ替えたindexを、ミニバッチの対象数（128件）毎にまわしていきます。
3.対象のミニバッチデータのそれぞれに、時系列データの50個分の過去データを付与します。
4.PyTorchのモデルを使って学習させます。
5.epoch毎に評価用データを使って予測、結果を確認します。
6.各評価用データの結果を比較し、ベストのモデルを保存します。
7.最後にベストのモデルでテスト用のデータを評価します。
"""
train_size = X_train.size(0) #学習用データの行数を取得
best_acc_score = 0
for epoch in range(settings.n_epocs):
    # 1. まずはtrainデータのindexをランダムに入れ替える。最初のtime_steps分は使わない。
    perm_idx = np.random.permutation(np.arange(settings.time_steps, train_size))
    # 2. batch size毎にperm_idxの対象のindexを取得
    for t_i in range(0, len(perm_idx), settings.batch_size):  # 0からperm_idxの行数になるまでbatch_size(128)を増分として繰り返す
        batch_idx = perm_idx[t_i:(t_i + settings.batch_size)] #t_i件目からbatch_size(128)行目を抜き出す
        # 3. LSTM入力用の時系列データの準備
        # featsは、次元が(128(batch_size), 50(time_steps), 5(Columns)）でX_trainのbatch_idxの各インデックスの値が設定されたもの
        feats = prep_feature_data(batch_idx, settings.time_steps, X_train, settings.feature_num, device)
        y_target = y_train[batch_idx]                #学習用データからbatch_idxを抜き出してy_targetとする
        # 4. pytorch LSTMの学習実施
        model.zero_grad()
        train_scores = model(feats) # batch size x time steps x feature_num
        loss = loss_function(train_scores, y_target.view(-1, 1))
        loss.backward()
        optimizer.step()

    # 5. validationデータの評価
    print('EPOCH: ', str(epoch), ' loss :', loss.item())
    with torch.no_grad():
        feats_val = prep_feature_data(np.arange(settings.time_steps, X_val.size(0)), settings.time_steps, X_val, settings.feature_num, device)
        val_scores = model(feats_val)
        tmp_scores = val_scores.view(-1).to('cpu').numpy()
        bi_scores = np.round(tmp_scores)
        acc_score = accuracy_score(y_val[settings.time_steps:], bi_scores)
        roc_score = roc_auc_score(y_val[settings.time_steps:], tmp_scores)
        # Val ACC = validationデータに対する精度
        # ROC AUC = AUC(Area under an ROC curve) AUCが1に近いほど性能が高いモデル        
        print('Val ACC Score :', acc_score, ' ROC AUC Score :', roc_score)

    # 6. validationの評価が良ければモデルを保存
    if acc_score > best_acc_score:
        best_acc_score = acc_score
        if not os.path.exists("./models"):
            os.mkdir("./models")
        #if os.path.exists('./models/pytorch_v1.mdl'):
        #    os.remove('./models/pytorch_v1.mdl')
        torch.save(model.state_dict(),'./models/pytorch_v1.mdl')
        print('best score updated, Pytorch model was saved!!', )

# 7. bestモデルで予測する。
model.load_state_dict(torch.load('./models/pytorch_v1.mdl'))
with torch.no_grad():
    feats_test = prep_feature_data(np.arange(settings.time_steps, X_test.size(0)), settings.time_steps, X_test, settings.feature_num, device)
    val_scores = model(feats_test)
    tmp_scores = val_scores.view(-1).to('cpu').numpy()
    bi_scores = np.round(tmp_scores)
    acc_score = accuracy_score(y_test[settings.time_steps:], bi_scores)
    roc_score = roc_auc_score(y_test[settings.time_steps:], tmp_scores)
    print('Test ACC Score :', acc_score, ' ROC AUC Score :', roc_score)
