import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import Variable

class RNN(chainer.Chain):
    def __init__(self, n_units, n_output):
        super(RNN, self).__init__()
        with self.init_scope():
            self.l1 = L.LSTM(None, n_units)
            self.l2 = L.Linear(None, n_output)

    def reset_state(self):
        self.l1.reset_state()

    def __call__(self, x, t):
        y = self.predict(x)
        loss = F.mean_squared_error(y, t)
        chainer.report({'loss': loss}, self)
        return loss

    def predict(self, x):
        h1 = F.dropout(self.l1(x), ratio=0.5)
        return self.l2(h1)

def main():
    start  = dt.date(2016, 1, 1)
    end = dt.date(2017, 9, 20)
    # 現状Yahoo Financeからデータを取れないのでmorningstarを使う
    df = web.DataReader('AMZN', 'morningstar', start, end)
    indexes = df.index.get_level_values('Date').tolist()
    values = np.array(df['Close'].tolist()).reshape(-1, 1)

    scaler = MinMaxScaler()
    values = scaler.fit_transform(values)[:, 0]

    xs, ts = [], []
    N = len(values)
    M = 25

    for n in range(M, N):
        x = values[n - M:n]
        t = values[n]
        xs.append(x)
        ts.append(t)

    # BUG: "pip install h5py==2.8.0rc1" でh5pyのRC版をインストールする必要あり
    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.float32).reshape(-1, 1)

    model = RNN(30, 1)
    serializers.load_npz('./rnn.model', model)
    model.reset_state()

    with chainer.using_config('train', False):
        ys = model.predict(Variable(xs)).data[:, 0]


    plt.figure(figsize=(12, 6))
    plt.plot(ys, color='red')
    plt.plot(ts[:, 0], color='blue')
    plt.show()

if __name__ == '__main__':
    main()