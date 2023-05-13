import lightgbm as lgbm
import numpy as np 
import pandas as pd 
import os 
import json
import ccxt
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.utils.class_weight import compute_class_weight
from random import randint


with open('BTC_USDT-4h-10.json') as f:
    d = json.load(f)
    
chart_df = pd.DataFrame(d)
chart_df = chart_df.rename(columns={0:"timestamp",
                                    1:"open",
                                    2:"high",
                                    3:"low",
                                    4:"close",
                                    5:"volume"})

def process(df): 
    binance = ccxt.binance() 
    dates = df['timestamp'].values 
    timestamp = [] 
    for i in range(len(dates)): 
        date_string = binance.iso8601(int(dates[i])) 
        date_string = date_string[:10] + " " + date_string[11:-5] 
        timestamp.append(date_string) 
    df['datetime'] = timestamp 
    df = df.drop(columns={'timestamp'})
    return df

chart_df = process(chart_df)
chart_df.tail() 

hours, days, months, years = [],[],[],[] 

for dt in tqdm(chart_df['datetime']):
    hour = pd.to_datetime(dt).hour 
    day = pd.to_datetime(dt).day 
    month = pd.to_datetime(dt).month 
    year = pd.to_datetime(dt).year 
    hours.append(hour) 
    days.append(day) 
    months.append(month)
    years.append(year) 

chart_df['hours'] = hours 
chart_df['days'] = days 
chart_df['months'] = months 
chart_df['years'] = years

chart_df.tail() 

targets = [] 
close = chart_df['close'].values 
high = chart_df['high'].values 
low = chart_df['low'].values 

threshold = 0.0075

for i in range(close.shape[0]-1): 
    high_volatility = (high[i+1] - close[i]) / close[i] 
    low_volatility = (low[i+1] - close[i]) / close[i] 
    if np.abs(high_volatility) >= np.abs(low_volatility): 
        if high_volatility >= threshold:
            targets.append(0) # long, tp=+0.75% 
        else:
            targets.append(2) # long, tp=+0.4% 
    elif np.abs(high_volatility) < np.abs(low_volatility):
        if low_volatility <= -threshold: 
            targets.append(1) # short, tp=-0.75%
        else:
            targets.append(3) # short, tp=-0.4%  
    
targets.append(None) 

chart_df['Targets'] = targets 
chart_df.head(5)

chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)

### addition of chart features ### 
chart_df["bop"] = chart_df.ta.bop(lookahead=False) 
chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False) 
chart_df["cmf"] = chart_df.ta.cmf(lookahead=False) 
chart_df["vwap"] = chart_df.ta.vwap(lookahead=False) 
chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100
chart_df["high/low"] = chart_df["high"] / chart_df["low"] 
chart_df["close/open"] = chart_df["close"] / chart_df["open"] 
chart_df["high/open"] = chart_df["high"] / chart_df["open"] 
chart_df["low/open"] = chart_df["low"] / chart_df["open"] 
chart_df["hwma"] = chart_df.ta.hwma(lookahead=False) 
chart_df["linreg"] = chart_df.ta.linreg(lookahead=False) 
chart_df["hwma/close"] = chart_df["hwma"] / chart_df["close"] 
chart_df["linreg/close"] = chart_df["linreg"] / chart_df["close"] 



### addition of moving average features ### 
windows = [5, 10, 20, 60, 120]
for window in windows:
    chart_df["close_ma{}".format(window)] = chart_df["close"].rolling(window).mean() 
    chart_df["close_ma{}_ratio".format(window)] = (chart_df["close"] - chart_df["close_ma{}".format(window)])/chart_df["close_ma{}".format(window)]    
    
### addition of recent differenced features ### 
for l in range(1, 6): 
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        val = chart_df[col].values 
        val_ret = [None for _ in range(l)]
        for i in range(l, len(val)):
            if val[i-l] == 0: 
                ret = 1 
            else:
                ret = val[i] / val[i-l]  
            val_ret.append(ret) 
        chart_df["{}_change_{}".format(col, l)] = val_ret
        
### drop unnecessary columns ### 
chart_df.drop(columns={"open","high","low","close","volume","vwap","hwma","linreg",
                       "close_ma5","close_ma10","close_ma20", "close_ma60","close_ma120"}, inplace=True) 


chart_df.dropna(inplace=True)

print(chart_df.shape)  

columns = chart_df.columns 

train_columns = [] 

for c in columns:
    if c not in ["years","datetime","Targets"]: 
        train_columns.append(c) 
        
        
train_idx = int(chart_df.shape[0] * 0.8) 
val_idx = int(chart_df.shape[0] * 0.1)
train_df, val_df, test_df = chart_df.iloc[:train_idx], chart_df.iloc[train_idx:train_idx+val_idx], chart_df.iloc[train_idx+val_idx:]


train_df.shape, val_df.shape, test_df.shape

chart_df.dropna(inplace=True)
X_train = train_df[train_columns] 
Y_train = train_df["Targets"]

X_val = val_df[train_columns] 
Y_val = val_df["Targets"] 

X_test = test_df[train_columns] 
Y_test = test_df["Targets"] 


class_weights = compute_class_weight(class_weight = "balanced",
                                     classes = np.unique(Y_train),
                                     y = Y_train) 

class_weights

params = {"objective":"multiclass", 
          "metric":"multi_logloss", 
          "num_class":4, 
          "boosting":"goss"} 

train_ds = lgbm.Dataset(X_train, label=Y_train) 
val_ds = lgbm.Dataset(X_val, label=Y_val) 
model = lgbm.train(params, train_ds, 1000, val_ds, verbose_eval=100) 

Y_pred = model.predict(X_test) 
classes = np.argmax(Y_pred, axis=1) 

cnt = 0 
for i in range(len(classes)):
    if classes[i] == Y_test.values[i]:
        cnt += 1 
        
cnt / len(classes) * 100 

import joblib
joblib.dump(model, 'lgb.pkl')