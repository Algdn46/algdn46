!pip install ccxt pandas numpy tensorflow sklearn python-telegram-bot python-dotenv

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv
import telegram
from telegram.ext import Updater, CommandHandler
import requests
from io import StringIO

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GitHub'dan gateio.env dosyasını çek
GITHUB_ENV_URL = "https://raw.githubusercontent.com/Algdn46/algdn46/main/gateio.env"
response = requests.get(GITHUB_ENV_URL)
if response.status_code != 200:
    raise FileNotFoundError("gateio.env dosyası GitHub'da bulunamadı veya erişilemedi!")

# .env içeriğini yükle
env_content = response.text
with open('gateio.env', 'w') as f:
    f.write(env_content)
load_dotenv('gateio.env')

# Gate.io API
exchange = ccxt.gate({
    'apiKey': os.getenv('GATEIO_API_KEY'),
    'secret': os.getenv('GATEIO_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

# Telegram Bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN or not os.getenv('GATEIO_API_KEY') or not os.getenv('GATEIO_SECRET_KEY'):
    raise ValueError("API anahtarları veya Telegram token eksik!")
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Global ayarlar
SYMBOLS = []
running = True
TIMEFRAMES = ['5m', '15m', '1h']
LEVERAGE = 10
RISK_PER_TRADE = 0.02
LSTM_LOOKBACK = 50
EMA_PERIODS = (9, 21)
RSI_PERIOD = 14
ATR_PERIOD = 14
chat_ids = set()
signals = {}
open_positions = {}
lstm_model = None
signals_lock = threading.Lock()
positions_lock = threading.Lock()

# LSTM Model
def create_lstm_model(output_units=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, 3)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def load_lstm_model():
    global lstm_model
    lstm_model = create_lstm_model()  # Colab'da model dosyası olmayacak, yeni oluşturuyoruz
    logging.info("Yeni LSTM modeli oluşturuldu.")

# Sembol Çekme
def get_all_futures_symbols():
    try:
        markets = exchange.load_markets()
        logging.info(f"Toplam piyasa: {len(markets)}")
        futures_symbols = [symbol for symbol, market in markets.items() if market.get('type') == 'swap' and market.get('active', True)]
        logging.info(f"Vadeli semboller: {len(futures_symbols)} adet")
        return futures_symbols
    except ccxt.NetworkError as e:
        logging.error(f"Sembol çekme ağ hatası: {e}")
        return []
    except ccxt.ExchangeError as e:
        logging.error(f"Sembol çekme borsa hatası: {e}")
        return []
    except Exception as e:
        logging.error(f"Sembol çekme bilinmeyen hata: {e}")
        return []

# Veri Toplama ve Sinyal Üretimi
def fetch_multi_tf_data(symbol):
    data = {}
    for tf in TIMEFRAMES:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['EMA9'] = df['close'].ewm(span=EMA_PERIODS[0]).mean()
            df['EMA21'] = df['close'].ewm(span=EMA_PERIODS[1]).mean()
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            df['RSI'] = 100 - (100 / (1 + gain.rolling(window=RSI_PERIOD).mean() / loss.rolling(window=RSI_PERIOD).mean()))
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=ATR_PERIOD).mean()
            data[tf] = df
        except ccxt.NetworkError as e:
            logging.error(f"{symbol} {tf} ağ hatası: {e}")
        except ccxt.ExchangeError as e:
            logging.error(f"{symbol} {tf} borsa hatası: {e}")
        except Exception as e:
            logging.error(f"{symbol} {tf} bilinmeyen hata: {e}")
    return data

def generate_signal(symbol):
    data = fetch_multi_tf_data(symbol)
    if not data or '5m' not in data or len(data['5m']) < LSTM_LOOKBACK:
        return
    try:
        current_price = exchange.fetch_ticker(symbol)['last']
        trend_filter = (data['1h']['EMA9'].iloc[-1] > data['1h']['EMA21'].iloc[-1]) and (data['15m']['EMA9'].iloc[-1] > data['15m']['EMA21'].iloc[-1])
        scaler = MinMaxScaler()
        recent_data = data['5m'][['close', 'high', 'low']].tail(LSTM_LOOKBACK)
        scaled_data = scaler.fit_transform(recent_data)
        lstm_input = scaled_data.reshape(1, LSTM_LOOKBACK, 3)
        if lstm_model is None:
            load_lstm_model()
        if lstm_model is None:
            logging.error(f"{symbol} için LSTM modeli yüklenemedi.")
            return
        pred_price = lstm_model.predict(lstm_input, verbose=0)[0][0]
        long_condition = (data['5m']['EMA9'].iloc[-2] < data['5m']['EMA21'].iloc[-2] and
                          data['5m']['EMA9'].iloc[-1] > data['5m']['EMA21'].iloc[-1] and
                          data['5m']['RSI'].iloc[-1] < 65 and trend_filter)
        short_condition = (data['5m']['EMA9'].iloc[-2] > data['5m']['EMA21'].iloc[-2] and
                           data['5m']['EMA9'].iloc[-1] < data['5m']['EMA21'].iloc[-1] and
                           data['5m']['RSI'].iloc[-1] > 35 and trend_filter)
        if not (long_condition or short_condition):
            return
        atr = data['5m']['ATR'].iloc[-1]
        tp_long, sl_long = current_price + 2 * atr, current_price - atr
        tp_short, sl_short = current_price - 2 * atr, current_price + atr
        balance = exchange.fetch_balance()['USDT']['free']
        risk = abs(current_price - sl_long) if long_condition else abs(sl_short - current_price)
        position_size = (balance * RISK_PER_TRADE) / risk if risk != 0 else 0

        with signals_lock:
            signals[symbol] = {
                'symbol': symbol, 'long': long_condition, 'short': short_condition,
                'entry': current_price, 'tp_long': tp_long, 'sl_long': sl_long,
                'tp_short': tp_short, 'sl_short': sl_short, 'size': position_size,
                'time': datetime.now().strftime('%H:%M:%S')
            }
        print(f"Sinyal: {symbol} - {'LONG' if long_condition else 'SHORT'} - Giriş: {current_price:.2f}")
    except Exception as e:
        logging.error(f"{symbol} sinyal hatası: {e}")

# Test için basit bir döngü
def test_run():
    global SYMBOLS
    SYMBOLS = get_all_futures_symbols()
    if not SYMBOLS:
        logging.error("Sembol listesi boş!")
        return
    print(f"Semboller yüklendi: {len(SYMBOLS)} adet")
    for symbol in SYMBOLS[:5]:  # İlk 5 sembolü test et
        generate_signal(symbol)
    print("Test tamamlandı.")

# Çalıştır
test_run()
