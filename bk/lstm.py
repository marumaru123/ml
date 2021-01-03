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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

future_num = 144　#何足先を予測するか
feature_num = 5 #volume, open, high, low, closeの5項目
batch_size = 128
time_steps = 50　# lstmのtimesteps
moving_average_num = 500　# 移動平均を取るCandle数
n_epocs = 30 
#データをtrain, testに分割するIndex
val_idx_from = 80000
test_idx_from = 100000

lstm_hidden_dim = 16
target_dim = 1

# 1. CSVファイルの読み込み
df = pd.read_csv('./data/USD_JPY_201601-201908_M10.csv', index_col='Datetime')

# 2. 教師データの作成
future_price = df.iloc[future_num:]['Close'].values
curr_price = df.iloc[:-future_num]['Close'].values
y_data_tmp = future_price - curr_price
y_data = np.zeros_like(y_data_tmp)
y_data[y_data_tmp > 0] = 1
y_data = y_data[moving_average_num:]
# 3. 価格の正規化
cols = df.columns
for col in cols:
    df['Roll_' + col] = df[col].rolling(window=500, min_periods=500).mean()
    df[col] = df[col] / df['Roll_' + col] - 1

#最初の500足分は移動平均データがないため除く。後半の144足分は予測データがないため除く
X_data = df.iloc[moving_average_num:-future_num][cols].values

# 4. データの分割、TorchのTensorに変換
#学習用データ
X_train = torch.tensor(X_data[:val_idx_from], dtype=torch.float, device=device)
y_train = torch.tensor(y_data[:val_idx_from], dtype=torch.float, device=device)
#評価用データ
X_val   = torch.tensor(X_data[val_idx_from:test_idx_from], dtype=torch.float, device=device)
y_val   = y_data[val_idx_from:test_idx_from]
#テスト用データ
X_test  = torch.tensor(X_data[test_idx_from:], dtype=torch.float, device=device)
y_test  = y_data[test_idx_from:]

class LSTMClassifier(nn.Module):

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

def prep_feature_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        # 過去のN足分をtime stepのデータとして格納する。
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]


    return feats

# Prepare for training
model = LSTMClassifier(feature_num, lstm_hidden_dim, target_dim).to(device)
loss_function = nn.BCELoss()
optimizer= optim.Adam(model.parameters(), lr=1e-4)

train_size = X_train.size(0)
best_acc_score = 0
for epoch in range(n_epocs):
    # 1. まずはtrainデータのindexをランダムに入れ替える。最初のtime_steps分は使わない。
    perm_idx = np.random.permutation(np.arange(time_steps, train_size))
    # 2. batch size毎にperm_idxの対象のindexを取得
    for t_i in range(0, len(perm_idx), batch_size):
        batch_idx = perm_idx[t_i:(t_i + batch_size)]
        # 3. LSTM入力用の時系列データの準備
        feats = prep_feature_data(batch_idx, time_steps, X_train, feature_num, device)
        y_target = y_train[batch_idx]
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
        print('Val ACC Score :', acc_score, ' ROC AUC Score :', roc_score)

    # 6. validationの評価が良ければモデルを保存
    if acc_score > best_acc_score:
        best_acc_score = acc_score
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
