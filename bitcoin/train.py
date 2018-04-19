from pathlib import Path
import pandas as pd
from sklearn import preprocessing

CSVPATH = Path('./data/btc.csv')

def replace_turnover(str):
  if 'K' in str:
    return float(str.replace('K', '')) * 1000
  elif 'M' in str:
    return float(str.replace('M', '')) * 1000000
  else:
    return str

def preprocess(csv_data):
  res = csv_data.sort_values(['日付け']).reset_index(drop=True)
  res['終値'] = res['終値'].map(lambda x: x.replace(',', '')).astype(float)
  res['終値'] = preprocessing.scale(res['終値'])
  res['始値'] = res['始値'].map(lambda x: x.replace(',', '')).astype(float)
  res['始値'] = preprocessing.scale(res['始値'])
  res['高値'] = res['高値'].map(lambda x: x.replace(',', '')).astype(float)
  res['高値'] = preprocessing.scale(res['高値'])
  res['安値'] = res['安値'].map(lambda x: x.replace(',', '')).astype(float)
  res['安値'] = preprocessing.scale(res['安値'])
  res['出来高'] = res['出来高'].map(lambda x: replace_turnover(x))
  res['出来高'] = preprocessing.scale(res['出来高'])
  res['前日比%'] = res['前日比%'].astype(float)
  res['前日比%'] = preprocessing.scale(res['前日比%'])
  return res.values.tolist()

def main():
  csv_data = pd.read_csv(CSVPATH)
  csv_data = preprocess(csv_data)
  print(csv_data)

if __name__ == '__main__':
  main()