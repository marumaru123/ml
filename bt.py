#!/usr/bin/env python
# coding: utf-8

# バックテスト
from backtesting import Strategy
from backtesting import Backtest
import pandas as pd
from lstmclassifier import LSTMClassifier
import torch
import numpy as np
import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

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
df = pd.read_csv('./data/data.csv', index_col='Datetime')
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
        self.model = LSTMClassifier(settings.feature_num, settings.lstm_hidden_dim, settings.target_dim).to(device)
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
        if len(self.data) >= settings.moving_average_num + settings.time_steps and len(self.data) % settings.future_num == 0:
            # 2. 推測用データの用意
            x_array = self.prepare_data()
            x_tensor = torch.tensor(x_array, dtype=torch.float, device=device)
            # 3. 予測の実行
            with torch.no_grad():
                y_pred = self.predict(x_tensor.view(1, settings.time_steps, settings.feature_num))

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
            tmp_df['Roll_' + col] = tmp_df[col].rolling(window=settings.moving_average_num, min_periods=settings.moving_average_num).mean()
            tmp_df[col] = tmp_df[col] / tmp_df['Roll_' + col] - 1

        #最後のtime_steps分のみの値を返す
        return tmp_df.tail(settings.time_steps)[cols].values

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
output = bt.run()
print(output)
