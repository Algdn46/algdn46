import ccxt
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# Config Yükleme
load_dotenv('config.env')

# API Konfigürasyonu
exchange = ccxt.mexc({
        'apiKey': os.getenv('API_KEY'),
        'secret': os.getenv('SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
})

# Global Ayarlar
SYMBOLS = []
TIMEFRAMES = ['5m', '15m', '1h']
LEVERAGE = 10
RISK_PER_TRADE = 0.02
LSTM_LOOKBACK = 50
EMA_PERIODS = (9, 21)
RSI_PERIOD = 14
ATR_PERIOD = 14

open_positions = {}
signals = {}
lstm_model = None

# Teknik Göstergeler
def add_indicators(df):
        # EMA
        df['EMA9'] = df['close'].ewm(span=9).mean()
        df['EMA21'] = df['close'].ewm(span=21).mean()

    # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bantları
        df['MA20'] = df['close'].rolling(20).mean()
        df['STD20'] = df['close'].rolling(20).std()
        df['UpperBand'] = df['MA20'] + 2*df['STD20']
        df['LowerBand'] = df['MA20'] - 2*df['STD20']

    # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(ATR_PERIOD).mean()

    return df

# Risk Yönetimi
def dynamic_risk(symbol, df):
        volatility = df['ATR'].iloc[-1] / df['close'].iloc[-1]
        if volatility > 0.05: return 0.01
elif volatility > 0.03: return 0.02
else: return 0.03

def calculate_size(entry, sl, risk):
        balance = exchange.fetch_balance()['USDT']['free']
        return (balance * risk) / abs(entry - sl)

# Bildirim Sistemi
def send_alert(message):
        token = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url)

# LSTM Modeli
def create_lstm():
        model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, 5)),
                    Dropout(0.3),
                    LSTM(32),
                    Dropout(0.3),
                    Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

def load_model():
        global lstm_model
        try:
                    lstm_model = load_model('lstm_model.h5')
                except:
        lstm_model = create_lstm()

# Veri Çekme
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data(symbol, timeframe):
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return add_indicators(df)

# Sinyal Üretme
def generate_signal(symbol):
        try:
                    df = fetch_data(symbol, '5m')
                    ticker = exchange.fetch_ticker(symbol)
                    price = ticker['last']

        # LSTM Tahmini
            scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['close','high','low','volume','ATR']].tail(LSTM_LOOKBACK))
        prediction = lstm_model.predict(scaled.reshape(1, LSTM_LOOKBACK, 5))
        pred_price = scaler.inverse_transform(prediction)[0][0]

        # Sinyal Koşulları
        macd_bullish = df['MACD'].iloc[-1] > df['Signal'].iloc[-1]
        in_lower = price < df['LowerBand'].iloc[-1]
        volume_spike = df['volume'].iloc[-1] > df['volume'].mean()

        long = macd_bullish and in_lower and volume_spike
        short = not macd_bus
