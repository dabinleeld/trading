
import time
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import telegram
from ccxt.base.errors import ExchangeError
from ccxt.binance import binance as BinanceExchange
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.utils.class_weight import compute_class_weight
from random import randint
import joblib
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import requests
from dateutil import parser
from binance.client import Client
from binance.helpers import round_step_size
from tqdm import tqdm 
class LGBM_Trader:
    stop_percent = 2.0
    big_vol_target_percent = 0.75
    small_vol_target_percent = 0.4

    def __init__(self, binance_cred, telegram_cred, model_ckpt):
        self.exchange: BinanceExchange = ccxt.binance({
            "enableRateLimit": True,
            "apiKey": binance_cred["api_key"],
            "secret": binance_cred["api_secret"]
        })
        # python-binance object
        self.client = Client(api_key=binance_cred["api_key"],
                             api_secret=binance_cred["api_secret"])
        self.model = joblib.load(model_ckpt)
        self.telebot = telegram.Bot(token=telegram_cred["token"])
        self.telegram_chat_id = telegram_cred["chat_id"]
        self.symbol = "BTCUSDT"
    

    def get_df(self):
        df = pd.DataFrame(self.exchange.fetch_ohlcv(self.symbol, timeframe="4h", limit=300))
        df = df.rename(columns={0: "timestamp",
                                1: "open",
                                2: "high",
                                3: "low",
                                4: "close",
                                5: "volume"})
        return df

    def create_timestamps(self, df):
        dates = df["timestamp"].values
        timestamp = []
        for i in range(len(dates)):
            date_string = self.exchange.iso8601(int(dates[i]))
            date_string = date_string[:10] + " " + date_string[11:-5]
            timestamp.append(date_string)
        df["datetime"] = timestamp
        df = df.drop(columns={"timestamp"})
        df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        return df

    def my_floor(self, a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    def get_tick_size(self, symbol: str) -> float:
        info = self.client.futures_exchange_info()
        for symbol_info in info['symbols']:
            if symbol_info['symbol'] == symbol:
                for symbol_filter in symbol_info['filters']:
                    if symbol_filter['filterType'] == 'PRICE_FILTER':
                        return float(symbol_filter['tickSize'])

    def get_rounded_price(self, symbol: str, price: float) -> float:
        return round_step_size(price, self.get_tick_size(symbol))

    @staticmethod
    def preprocess_data(chart_df):
        close_prices = chart_df["close"].values

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
                            "close_ma5","close_ma10","close_ma20", "close_ma60","close_ma120",
                            "years","datetime"}, inplace=True)

        chart_df.dropna(inplace=True)
        return chart_df, close_prices

    def send_message(self, text):
        try:
            self.telebot.sendMessage(chat_id=self.telegram_chat_id, text=text)
        except Exception as e:
            print(e)

    def get_position_size(self):
        positions = self.client.futures_account()["positions"]
        for p in positions:
            if p["symbol"] == self.symbol:
                amt = p["positionAmt"]
                return np.abs(float(amt))
            
    def get_best_bid_ask(self):
        orderbook = self.client.get_order_book(symbol=self.symbol)
        max_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        min_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
        return max_bid, min_ask

    def place_best_buy_limit_order(self, reduce_only, qty, stopPrice, targetPrice):
        futures_order = self.client.futures_create_order(
            symbol=self.symbol,
            side="BUY",
            quantity=qty,
            reduceOnly=reduce_only,
            type="MARKET")
        if reduce_only == False: # send in stop loss and take profit
            futures_stop_loss = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="SELL",
                type="STOP_MARKET",
                stopPrice=stopPrice,
                closePosition=True)
            futures_take_profit = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="SELL",
                type="TAKE_PROFIT_MARKET",
                stopPrice=targetPrice,
                closePosition=True)
            stoploss_id = futures_stop_loss['orderId']
            takeprofit_id = futures_take_profit['orderId']
            return stoploss_id, takeprofit_id
        else:
            return None, None
    def place_best_sell_limit_order(self, reduce_only, qty, stopPrice, targetPrice):
        futures_order = self.client.futures_create_order(
            symbol=self.symbol,
            side="SELL",
            quantity=qty,
            reduceOnly=reduce_only,
            type="MARKET")
        if reduce_only == False: # send in take profit and stop loss
            futures_stop_loss = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="BUY",
                type="STOP_MARKET",
                stopPrice=str(stopPrice),
                closePosition=True)
            futures_take_profit = self.client.futures_create_order(
                symbol=self.symbol,
                timeInForce="GTC",
                side="BUY",
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(targetPrice),
                closePosition=True)
            stoploss_id = futures_stop_loss['orderId']
            takeprofit_id = futures_take_profit['orderId']
            return stoploss_id, takeprofit_id
        else:
            return None, None

    def execute_trade(self):
        # run trade in 4 hour cycles (1 -> 5 -> 9 -> 1 -> ...)
        iteration = 0
        move = 0 # -1: short, 1: long
        stoploss_id, takeprofit_id = -1, -1 # stop loss and take profit id, close them after a single iteration just in case they are not filled
        while True:
            self.send_message("========== Trade Iteration {} ==========".format(iteration))
            t0 = time.time()
            df = self.get_df()
            df = self.create_timestamps(df)

            df, close = self.preprocess_data(df)
            x = df.values[-2].reshape((-1, df.shape[1]))
            pred = self.model.predict(x)
            pred_class = np.argmax(pred, axis=1)
            prev_close = close[-2]

            pred = pred_class[0]
            pos_dict = {0:"Long", 1:"Short", 2:"Weak Long", 3:"Weak Short"}
            self.send_message("LGBM Directional Prediction {}".format(pos_dict[pred]))
            if iteration > 0:
                # get rid of unclosed take profit and stop loss orders
                try:
                    self.client.futures_cancel_order(symbol=self.symbol, orderId=stoploss_id)
                except Exception as e:
                    print(e)
                try:
                    self.client.futures_cancel_order(symbol=self.symbol, orderId=takeprofit_id)
                except Exception as e:
                    print(e)
                # close if there are any positions open from the previous iteration
                qty = self.get_position_size()
                print("quantity = {}".format(qty))
                if qty == 0:
                    self.send_message("no positions open... stop loss or take profit was probably triggered.")
                else:
                    if move == -1:
                        self.send_message("Closing previous short position...")
                        self.place_best_buy_limit_order(reduce_only=True, qty=qty, stopPrice=None, targetPrice=None)
                    elif move == 1:
                        self.send_message("Closing previous long position...")
                        self.place_best_sell_limit_order(reduce_only=True, qty=qty, stopPrice=None, targetPrice=None)

            if pred == 0 or pred == 2:
                btc_usdt = self.client.get_symbol_ticker(symbol=self.symbol)
                btc_usdt = float(btc_usdt['price'])
                stopPrice = prev_close * (1 - self.stop_percent / 100)
                if pred == 0:
                    targetPrice = prev_close * (1 + self.big_vol_target_percent / 100)
                elif pred == 2:
                    targetPrice = prev_close * (1 + self.small_vol_target_percent / 100)
                stopPrice = self.get_rounded_price(self.symbol, stopPrice)
                targetPrice = self.get_rounded_price(self.symbol, targetPrice)
                usdt = self.client.futures_account_balance()[6]["balance"] # get usdt balance
                self.send_message("current cash status = {}".format(usdt))
                qty = float(usdt) / float(btc_usdt)
                qty = self.my_floor(qty, precision=3)
                stoploss_id, takeprofit_id = self.place_best_buy_limit_order(reduce_only=False,
                                                                            qty=qty,
                                                                            stopPrice=stopPrice,
                                                                            targetPrice=targetPrice)
                move = 1
            elif pred == 1 or pred == 3:
                btc_usdt = self.client.get_symbol_ticker(symbol=self.symbol)
                btc_usdt = float(btc_usdt['price'])
                stopPrice = prev_close * (1 + self.stop_percent / 100)
                if pred == 1:
                    targetPrice = prev_close * (1 - self.big_vol_target_percent / 100)
                elif pred == 3:
                    targetPrice = prev_close * (1 - self.small_vol_target_percent / 100)
                stopPrice = self.get_rounded_price(self.symbol, stopPrice)
                targetPrice = self.get_rounded_price(self.symbol, targetPrice)
                usdt = self.client.futures_account_balance()[6]["balance"] # get usdt balance
                self.send_message("current cash status = {}".format(usdt))
                qty = float(usdt) / float(btc_usdt)
                qty = self.my_floor(qty, precision=3)
                stoploss_id, takeprofit_id = self.place_best_sell_limit_order(reduce_only=False,
                                                                            qty=qty,
                                                                            stopPrice=stopPrice,
                                                                            targetPrice=targetPrice)
                move = -1
            iteration += 1
            self.send_message("waiting for the next 4 hours...")
            elapsed = time.time() - t0
            time.sleep(60*60*4 - elapsed)

### run trade ###
binance_cred = {
    "api_key": "Rpr5UtFkf9W4bHEHUgWqt5pHXmkbXske65vEytspyTyHdffamfxZDEdlb9XOI9kX",
    "api_secret": "RRcdHm5IITSMtctsX4mW1ugFTPouNAV9i1K54Couh88fw9KQ6v43LjOtrE153uVC"
}
telegram_cred = {
    "token": "6224942923:AAG2vGGT4p5KUGkv9iKrCHSYTvjFLmdZaOk",
    "chat_id": "63604369" 
}
trader = LGBM_Trader(binance_    = binance_cred,telegram_cred = telegram_cred,model_ckpt = "lgb.pkl")
# 한국 시간으로 1시, 5시, 9시, 13시 ,17시, 21시 중 하나에 시작해야함
trader.execute_trade()