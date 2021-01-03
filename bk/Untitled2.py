#!/usr/bin/env python
# coding: utf-8

# In[9]:


from datetime import datetime

print("・unixtime⇒datetimeに変換")
print(datetime.fromtimestamp(1519862400))

print("・datetime⇒unixtimeに変換")
print(datetime(2018,3,1,0,0).strftime('%s'))


# In[1]:


"""
PyTorchでMNISTを学習させる

:summary   PyTorchで単純な多層パーセプトロンを構築してみる
:author    RightCode Inc. (https://rightcode.co.jp)
"""

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)

        return f.log_softmax(x, dim=1)


def load_MNIST(batch=128, intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch,
        shuffle=True)

    return {'train': train_loader, 'test': test_loader}


if __name__ == '__main__':
    # 学習回数
    epoch = 20

    # 学習結果の保存用
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }

    # ネットワークを構築
    net: torch.nn.Module = MyNet()

    # MNISTのデータローダーを取得
    loaders = load_MNIST()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    for e in range(epoch):

        """ Training Part"""
        loss = None
        # 学習開始 (再開)
        net.train(True)  # 引数は省略可能
        for i, (data, target) in enumerate(loaders['train']):
            # 全結合のみのネットワークでは入力を1次元に
            # print(data.shape)  # torch.Size([128, 1, 28, 28])
            data = data.view(-1, 28*28)
            # print(data.shape)  # torch.Size([128, 784])

            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Training log: {} epoch ({} / 60000 train. data). Loss: {}'.format(e+1,
                                                                                         (i+1)*128,
                                                                                         loss.item())
                      )

        history['train_loss'].append(loss)

        """ Test Part """
        # 学習のストップ
        net.eval()  # または net.train(False) でも良い
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loaders['test']:
                data = data.view(-1, 28 * 28)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= 10000

        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                         correct / 10000))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct / 10000)

    # 結果の出力と描画
    print(history)
    plt.figure()
    plt.plot(range(1, epoch+1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epoch+1), history['test_loss'], label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(range(1, epoch+1), history['test_acc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.savefig('test_acc.png')


# In[260]:


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


# In[269]:


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
future_num = 96 # 翌日の同時間帯がどれだけ後か 15分足だと1440 / 15 = 96

#volume, open, high, low, closeの5項目
feature_num = 5 #特徴量

batch_size = 128 # LSTMに渡すレコード数のこと

# lstmのtimesteps
time_steps = 50

# 移動平均を取るCandle数
moving_average_num = 500
#moving_average_num = 300

#n_epocs = 30
n_epocs = 30 #学習する回数

#データをtrain, testに分割するIndex
#val_idx_from = 80000
#test_idx_from = 100000
val_idx_from  = 80000   # csvのこれより前のレコードを学習に使用する
test_idx_from = 100000  # csvのval_idx_fromより後で、これより前のレコードを評価に使用して、これより後のレコードをテスト用データとする

lstm_hidden_dim = 16 # LSTMの隠れ層の出力サイズ
target_dim = 1       # LSTMの隠れ層の最終出力サイズ

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
df = pd.read_csv('./data/binance/data.csv', index_col='Datetime')

# 2. 教師データの作成
future_price = df.iloc[future_num:]['Close'].values #dfのうちfuture_num個目以降のCloseをfuture_priceとする
curr_price = df.iloc[:-future_num]['Close'].values  #dfのうち後ろからfuture_num個目のCloseをcurr_priceとする
y_data_tmp = future_price - curr_price              # future_price - curr_price
y_data = np.zeros_like(y_data_tmp)                  # 教師データをゼロで初期化
y_data[y_data_tmp > 0] = 1                          # future_price - curr_priceがゼロより大きい場合1とする。
y_data = y_data[moving_average_num:]                # 移動平均を算出する行以降を最終的な教師データとする
# 3. 価格の正規化
cols = df.columns
for col in cols:
    df['Roll_' + col] = df[col].rolling(window=500, min_periods=500).mean()
    #df['Roll_' + col] = df[col].rolling(window=300, min_periods=300).mean() # 移動平均を算出してRoll_[カラム名]というカラムを作成して設定する
    df[col] = df[col] / df['Roll_' + col] - 1                           # カラム / 移動平均 - 1

#最初の500足分は移動平均データがないため除く。後半の144足分は予測データがないため除く
X_data = df.iloc[moving_average_num:-future_num][cols].values

# 4. データの分割、TorchのTensorに変換
#学習用データ(X_dataとy_dataのインデックスがval_idx_from件数までを学習用データとする)
X_train = torch.tensor(X_data[:val_idx_from], dtype=torch.float, device=device)
y_train = torch.tensor(y_data[:val_idx_from], dtype=torch.float, device=device)
#評価用データ(X_dataとy_dataのインデックスがval_idx_from件～test_idx_from件までを評価用データとする)
X_val   = torch.tensor(X_data[val_idx_from:test_idx_from], dtype=torch.float, device=device)
y_val   = y_data[val_idx_from:test_idx_from]
#テスト用データ(X_dataとy_dataのインデックスがtest_idx_from件以降をテスト用データとする)
X_test  = torch.tensor(X_data[test_idx_from:], dtype=torch.float, device=device)
y_test  = y_data[test_idx_from:]

"""
LSTMモデル定義
時系列データを処理するためのLSTMのクラスを定義します。

このクラスでは、（バッチ数、時系列データ数、特徴量数）のデータを受けて、
LSTMを通し、LSTMの最終出力をLinear層に渡し、Linear層の出力をsigmoidでバイナリの予測として出力する、というモデルにしています。

今回はLSTMで二値分類のため、LSTMの時系列の出力は利用せず、最終出力のみを利用します。
各データポイント毎に50個分の時系列データをLSTMに渡して、LSTMは50個分の時系列の結果を返しますが、
途中の結果は利用せずに最終出力結果のみを利用します。
"""
class LSTMClassifier(nn.Module):

    # lstm_input_dim  : 特徴量(カラム数5)
    # lstm_hidden_dim : LSTMの隠れ層の出力サイズ (16)
    # target_dim      : LSTMの隠れ層の最終出力サイズ (1)
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(LSTMClassifier, self).__init__()
        self.input_dim = lstm_input_dim
        self.hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=1, #default
                            #dropout=0.2,
                            batch_first=True
                            )
        self.dense = nn.Linear(lstm_hidden_dim, target_dim)

    def forward(self, X_input):
        _, lstm_out = self.lstm(X_input)
        # LSTMの最終出力のみを利用する。
        linear_out = self.dense(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)

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
model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
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
for epoch in range(n_epocs):
    # 1. まずはtrainデータのindexをランダムに入れ替える。最初のtime_steps分は使わない。
    perm_idx = np.random.permutation(np.arange(time_steps, train_size))
    # 2. batch size毎にperm_idxの対象のindexを取得
    for t_i in range(0, len(perm_idx), batch_size):  # 0からperm_idxの行数になるまでbatch_size(128)を増分として繰り返す
        batch_idx = perm_idx[t_i:(t_i + batch_size)] #t_i件目からbatch_size(128)行目を抜き出す
        # 3. LSTM入力用の時系列データの準備
        # featsは、次元が(128(batch_size), 50(time_steps), 5(Columns)）でX_trainのbatch_idxの各インデックスの値が設定されたもの
        feats = prep_feature_data(batch_idx, time_steps, X_train, feature_num, device)
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
        feats_val = prep_feature_data(np.arange(time_steps, X_val.size(0)), time_steps, X_val, feature_num, device)
        val_scores = model(feats_val)
        tmp_scores = val_scores.view(-1).to('cpu').numpy()
        bi_scores = np.round(tmp_scores)
        acc_score = accuracy_score(y_val[time_steps:], bi_scores)
        roc_score = roc_auc_score(y_val[time_steps:], tmp_scores)
        # Val ACC = validationデータに対する精度
        # ROC AUC = AUC(Area under an ROC curve) AUCが1に近いほど性能が高いモデル        
        print('Val ACC Score :', acc_score, ' ROC AUC Score :', roc_score)

    # 6. validationの評価が良ければモデルを保存
    if acc_score > best_acc_score:
        best_acc_score = acc_score
        #if os.path.exists('./models/pytorch_v1.mdl'):
        #    os.remove('./models/pytorch_v1.mdl')
        torch.save(model.state_dict(),'./models/pytorch_v1.mdl')
        print('best score updated, Pytorch model was saved!!', )

# 7. bestモデルで予測する。
model.load_state_dict(torch.load('./models/pytorch_v1.mdl'))
with torch.no_grad():
    feats_test = prep_feature_data(np.arange(time_steps, X_test.size(0)), time_steps, X_test, feature_num, device)
    val_scores = model(feats_test)
    tmp_scores = val_scores.view(-1).to('cpu').numpy()
    bi_scores = np.round(tmp_scores)
    acc_score = accuracy_score(y_test[time_steps:], bi_scores)
    roc_score = roc_auc_score(y_test[time_steps:], tmp_scores)
    print('Test ACC Score :', acc_score, ' ROC AUC Score :', roc_score)


# In[270]:


# バックテスト
from backtesting import Strategy
from backtesting import Backtest

"""
バックテスト用データの準備
バックテストに必要な時系列の価格データを用意します。
公式ドキュメントより、次の条件を満たす必要があります。

pandas DataFrame形式であること
'Open', 'High', 'Low', 'Close', 列を持っている事。（'Volume'はオプション）
indexはpandasのDatetime形式であること。
他の列は存在していても問題なし。
前回の記事 でデータを取得していれば、そのデータを使う事ができます。

以下ではcsvファイルを読み込んでpandasのDataFrameに格納していますが、
csvファイルを読み込んだ場合は明示的にpd.to_datetime(df.index)でindex列をDatetime型に設定しておく必要があります。
"""

df = pd.read_csv('./data/binance/data.csv', index_col='Datetime')
df.index = pd.to_datetime(df.index)

"""
Custom Strategyの作成
各データポイントでbuy/sellの指示を出すためのクラスを作成します。
backtestingのStrategyクラスを継承して、init()とnext()メソッドを実装する必要があります。
"""
class myLSTMStrategy(Strategy):

    """
    init() メソッド: 最初に呼び出されるので、効率化のためにテクニカルの指標等を事前に計算しておく事が推奨されています。
    """
    def init(self):
        # 1. LSTMの学習済みモデルの読み込み
        self.model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
        # load model
        self.model.load_state_dict(torch.load('./models/pytorch_v1.mdl'))

    """
    next() メソッド: こちらがメインのメソッドで、Backtestingから各データポイント毎に呼び出されます。
    各データポイントに対するbuyやsellの行動は、次のデータポイントのOpen値に対して実行されます。
    
    価格データの確認
    Backtestingでは、テストデータの各行（データポイント）を順に呼び出していきますが、
    その時にnext()メソッドがコールされます。

    そのnext()メソッド内では、そのデータポイント以前の価格データを参照する事ができますが、
    そのデータポイントより未来の価格データは確認する事ができません。
    つまり、next()メソッドの中においては、過去のデータから未来を予測する必要があります。

    注文の指示
    nextメソッドの中でbuyやsellの指示を出す事ができます。self.buy()、self.sell()をコールするだけです。引数で価格やロスカットポイントなどを指定できます。

    buy : 現在のポジションをクローズし、全力（予算全額）で買います。引数で購入価格（未設定の場合は成行）、損切値、利確値を指定できます。
    sell : buyの逆です。
    他にもPositionのclose()メソッドでポジションをクローズする事もできます。

    ここで、Close[-1]やOpen[-1]は、そのデータポイントの時点で一番最後の（最新の）Close値、Open値となります。
    機械学習モデルでは、ここで価格データやその時点で手に入る様々なデータをモデルに渡して推論し、その結果に応じて売買を行うロジックを書くという事になるでしょう。
    強化学習を使えば、強化学習から売買の指示を出す事も可能かと思います。
    """
    def next(self): 
        # 過去500ステップ分のデータが貯まるまではスキップ
        # 1日に1回のみ取引するため、hour & minuteが0の時のみ処理するようにする。
        if len(self.data) >= moving_average_num + time_steps and len(self.data) % future_num == 0:
            # 2. 推測用データの用意
            x_array = self.prepare_data()
            x_tensor = torch.tensor(x_array, dtype=torch.float, device=device)
            # 3. 予測の実行
            with torch.no_grad():
                y_pred = self.predict(x_tensor.view(1, time_steps, feature_num))

            # 4. 予測が買い(1)であればbuy()、それ以外はsell()
            if y_pred == 1:
                self.buy(sl=self.data.Close[-1]*0.99, 
                         tp=self.data.Close[-1]*1.01)
            else:
                self.sell(sl=self.data.Close[-1]*1.01, 
                         tp=self.data.Close[-1]*0.99)

    def prepare_data(self):
        # いったんPandasのデータフレームに変換
        tmp_df = pd.concat([
                    self.data.Volume.to_series(), 
                    self.data.Open.to_series(), 
                    self.data.High.to_series(), 
                    self.data.Low.to_series(), 
                    self.data.Close.to_series(), 
                    ], axis=1)

        # 500足の移動平均に対する割合とする。
        cols = tmp_df.columns
        for col in cols:
            tmp_df['Roll_' + col] = tmp_df[col].rolling(window=moving_average_num, min_periods=moving_average_num).mean()
            tmp_df[col] = tmp_df[col] / tmp_df['Roll_' + col] - 1

        #最後のtime_steps分のみの値を返す
        return tmp_df.tail(time_steps)[cols].values

    def predict(self, x_array):
        y_score = self.model(x_array) 
        return np.round(y_score.view(-1).to('cpu').numpy())[0]

"""
バックテストの実行
データとstrategyが用意できたので、早速バックテストを実行してみましょう。
バックテストの実行は非常に簡単です。

Backtestクラスのインスタンスを作成して、run()を呼ぶだけです。
コンストラクタの引数には、テストで利用するデータ（df）と、上記で作成したcustom strategyクラス、cash（予算）、comission（売買手数料）を指定します。

cashは10万円を、手数料はFXでは標準的な0.4銭に近い0.00004としました。

実行結果で主要な値を簡単に解説します。

項目	意味
Start, End	バックテスト対象期間の開始日、終了日
Duration	バックテストの期間。ここでは328日分が対象となっていました。
Exposure[%]	資産の変動率といった意味でしょうか。
Equity Final [$]	最終的に資産がどうなったか。ここでは10万円が10万906円となった事を意味します。若干プラスとなっていますね
Equity Peak [$]	一番儲かっていた時の資産額。ここでは10万6208円がピークでした。
Return [%]	リターン率。これが一番需要ですね。0.9%の増でした
Win Rate [%]	取引の勝率です。48.2456%と五分を若干下回る戦績でした。
Best Trade [%]	一番良い取引の率
Worst Trade [%]	一番悪かった取引の率
Avg Trade [%]	平均のリターン率
これらの情報を見ることで、アルゴリズムの性能の評価が可能となります。
また、plot()メソッドの呼び出しでいい感じのインタラクティブグラフを出力してくれます。
  bt.plot()

"""
bt = Backtest(df[1000:], myLSTMStrategy, cash=100000, commission=.00004)
bt.run()


# In[158]:


# cat *.csv > test.txt
# cat test.txt | sort > data.csv
# vi で見出しを消す
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

result = []
_endTime = 0
#for x in range(3):
while True:
    test = klines(endTime=_endTime)
    df = pd.DataFrame(test[1:])
    result = test[1:]
    #file_name = result[0][0].strftime('%Y%m%d%H%M%S') + "-" + result[len(result) - 1][0].strftime('%Y%m%d%H%M%S') + ".csv"
    file_name = result[0][0].strftime('%Y%m%d%H%M') + "-" + result[len(result) - 1][0].strftime('%Y%m%d%H%M') + ".csv"
    df = pd.DataFrame(result)
    df.columns = ['Datetime', 'Volume', 'Open', 'High', 'Low', 'Close']
    df = df.set_index('Datetime')
    df.to_csv('./data/binance/' + file_name)
    _endTime = int(test[0][0].timestamp() * 1000)


# In[ ]:




