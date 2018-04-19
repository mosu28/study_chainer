import pandas as pd
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt

def main():
    start  = dt.date(2016, 1, 1)
    end = dt.date(2017, 9, 20)
    # 現状Yahoo Financeからデータを取れないのでmorningstarを使う
    df = web.DataReader('AMZN', 'morningstar', start, end)
    indexes = df.index.get_level_values('Date').tolist()
    values = df['Close'].tolist()
    plt.plot(indexes, values)
    plt.show()

if __name__ == '__main__':
    main()
