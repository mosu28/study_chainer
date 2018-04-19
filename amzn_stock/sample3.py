import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import chainer

def main():
    start  = dt.date(2016, 1, 1)
    end = dt.date(2017, 9, 20)
    # 現状Yahoo Financeからデータを取れないのでmorningstarを使う
    df = web.DataReader('AMZN', 'morningstar', start, end)
    # indexes = df.index.get_level_values('Date').tolist()
    values = np.array(df['Close'].tolist()).reshape(-1, 1)
    scaler = MinMaxScaler()
    values = scaler.fit_transform(values)[:, 0]

    xs, ts = [], []
    N = len(values)
    M = 25

    # zip([1,2,3], [4,5,6]) => [[1, 4], [2, 5], [3, 6]]

    for n in range(M, N):
        # print('{}~{}: {}'.format(n-M, n, values[n]))
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


if __name__ == '__main__':
    main()