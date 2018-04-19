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

class LSTMUpdater(training.StandardUpdater):
    def __init__(self, data_iter, optimizer, device=None):
        super(LSTMUpdater, self).__init__(data_iter, optimizer, device=None)
        self.device = device

    def update_core(self):
        data_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = data_iter.__next__()
        x_batch, t_batch = chainer.dataset.concat_examples(batch, self.device)

        optimizer.target.reset_state()
        optimizer.target.cleargrads()
        loss = optimizer.target(x_batch, t_batch)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

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

    n_train = int(len(xs) * 0.6)
    dataset = list(zip(xs, ts))
    train, test = chainer.datasets.split_dataset(dataset, n_train)

    np.random.seed(1)

    model = RNN(30, 1)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    batchsize = 20
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    updater = LSTMUpdater(train_iter, optimizer, device=-1)

    epoch = 3000
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))

    trainer.run()

    # 学習モデルの保存
    serializers.save_npz('rnn.model', model)


if __name__ == '__main__':
    main()