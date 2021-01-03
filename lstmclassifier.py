#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn

# LSTM機械学習
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
