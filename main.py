import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
import requests
import ssl
from urllib3.util.ssl_ import create_urllib3_context
from ccxt import binance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Özel TLS yapılandırması
context = create_urllib3_context(ssl_minimum_version=ssl.TLSVersion.TLSv1_2)
context.set_ciphers("DEFAULT")
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount("https://", adapter)
session.verify = True

# Binance exchange nesnesini oluştur
exchange = binance({
    "session": session,
    "enableRateLimit": True,
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'options': {'defaultType': 'future'},
})

# Config ve Log Ayarları
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Sabitler
INTERVAL = '5m'
RISK_RATIO = 0.2  # Risk oranı %0.2
LOOKBACK = 60  # LSTM için geçmiş veri uzunluğu
last_signals = {}  # Tekrarlanan sinyalleri önlemek için

# Türkiye saati (UTC+3)
TR_TIMEZONE = timezone(timedelta(hours=3))

# Verileri ölçeklendirmek için scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# LSTM Modelini oluştur
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Fiyat tahmini için tek bir çıkış
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Veriyi hazırla ve modeli eğit
def train_lstm_model(symbol):
    try:
        # Geçmiş 1 yıllık 5 dakikalık veriyi çek
        ohlcv = exchange.fetch_ohlcv(symbol, INTERVAL, limit=10000)  # Yaklaşık 1 yıl
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
        
        # Sadece kapanış fiyatlarını kullan
        data = df['close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        
        # Eğitim verisi oluştur
        X_train, y_train = [], []
        for i in range(LOOKBACK, len(scaled_data)):
            X_train.append(scaled_data[i-LOOKBACK:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Modeli oluştur ve eğit
        model = create_lstm_model()
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
        return model
    except Exception as e:
        logger.error(f"LSTM modeli eğitme hatası: {str(e)}", exc_info=True)
        return None

async def generate_signal(symbol, model):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, INTERVAL, limit=LOOKBACK+1)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
        
        # Kapanış fiyatlarını al ve ölçeklendir
        data = df['close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        
        # Son 60 veriyi al ve tahmini yap
        X_test = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        predicted_price = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        
        last = df.iloc[-1]
        current_price = last['close']
        
        # ATR ile SL ve TP hesapla
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = ranges.rolling(window=14).mean().iloc[-1]
        
        # Sinyal üret
        if predicted_price > current_price * 1.005:  # %0.5 artış bekleniyorsa
            sl = last['low'] - (atr * RISK_RATIO)
            tp1 = last['close'] + (atr * RISK_RATIO * 1.0)
            tp2 = last['close'] + (atr * RISK_RATIO * 1.5)
            tp3 = last['close'] + (atr * RISK_RATIO * 2.0)
            return 'LONG', last['close'], sl, (tp1, tp2, tp3)
        elif predicted_price < current_price * 0.995:  # %0.5 düşüş bekleniyorsa
            sl = last['high'] + (atr * RISK_RATIO)
            tp1 = last['close'] - (atr * RISK_RATIO * 1.0)
            tp2 = last['close'] - (atr * RISK_RATIO * 1.5)
            tp3 = last['close'] - (atr * RISK_RATIO * 2.0)
            return 'SHORT', last['close'], sl, (tp1, tp2, tp3)
        
        return None, None, None, None
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}", exc_info=True)
        return None, None, None, None

async def format_telegram_message(symbol, direction, entry, sl, tp):
    try:
        clean_symbol = symbol.replace(':USDT-', '/USDT').split('/')[0] + '/USDT'
        color = '<span style="color:green">Long</span>' if direction == 'LONG' else '<span style="color:red">Short</span>'
        tp1, tp2, tp3 = tp
        message = f"""
🚦✈️ {clean_symbol} {color}
━━━━━━━━━━━━━━
🪂 Giriş: {entry:.3f}
🚫 SL: {sl:.3f}
🎯 TP1: {tp1:.3f}
🎯 TP2: {tp2:.3f}
🎯 TP3: {tp3:.3f}
🕒 Zaman: {datetime.now(TR_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}
"""
        return message
    except Exception as e:
        logger.error(f"Mesaj formatlama hatası: {str(e)}", exc_info=True)
        return "Mesaj formatlama hatası oluştu!"

async def scan_symbols(context: ContextTypes.DEFAULT_TYPE, chat_id: int, models: dict):
    try:
        logger.info("Sinyaller taranıyor...")
        for attempt in range(3):
            try:
                markets = exchange.load_markets()
                break
            except Exception as e:
                logger.error(f"load_markets attempt {attempt + 1} failed: {str(e)}", exc_info=True)
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    await context.bot.send_message(chat_id=chat_id, text="Binance verileri yüklenemedi, tekrar dene!")
                    return
        
        symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
        
        found_signal = False
        for symbol in symbols:
            try:
                if symbol not in models:
                    models[symbol] = train_lstm_model(symbol)
                model = models[symbol]
                if model is None:
                    continue
                direction, entry, sl, tp = await generate_signal(symbol, model)
                if direction and entry:
                    current_signal = (direction, entry, sl, tp)
                    if symbol not in last_signals or last_signals[symbol] != current_signal:
                        message = await format_telegram_message(symbol, direction, entry, sl, tp)
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML'
                        )
                        last_signals[symbol] = current_signal
                        found_signal = True
                        time.sleep(1)
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {str(e)}", exc_info=True)
        
        if not found_signal:
            await context.bot.send_message(chat_id=chat_id, text="Sinyal bulunamadı ede. Az sabret.")
    except Exception as e:
        logger.error(f"Genel tarama hatası: {str(e)}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text="Bir hata oluştu, tekrar dene!")

async def continuous_scan(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.bot_data.get('chat_id')
    models = context.bot_data.get('models', {})
    while True:
        try:
            logger.info("Sürekli sinyal tarama başlıyor...")
            markets = exchange.load_markets()
            symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
            found_signal = False
            for symbol in symbols:
                if symbol not in models:
                    models[symbol] = train_lstm_model(symbol)
                model = models[symbol]
                if model is None:
                    continue
                direction, entry, sl, tp = await generate_signal(symbol, model)
                if direction and entry:
                    current_signal = (direction, entry, sl, tp)
                    if symbol not in last_signals or last_signals[symbol] != current_signal:
                        message = await format_telegram_message(symbol, direction, entry, sl, tp)
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML'
                        )
                        last_signals[symbol] = current_signal
                        found_signal = True
                        time.sleep(1)
            if not found_signal:
                logger.info("Sinyal bulunamadı, 60 saniye bekleniyor...")
            context.bot_data['models'] = models
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Sürekli tarama hatası: {str(e)}", exc_info=True)
            await asyncio.sleep(60)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    context.bot_data['chat_id'] = chat_id
    context.bot_data['models'] = {}
    await update.message.reply_text("🚀 Kemerini tak dostum, sinyaller geliyor...")
    await scan_symbols(context, chat_id, context.bot_data['models'])
    context.job_queue.run_repeating(continuous_scan, interval=60, first=5)

def main():
    load_dotenv("config.env")
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN bulunamadı!")
        exit(1)

    try:
        application = Application.builder().token(token).build()
        application.add_handler(CommandHandler("start", start))
        logger.info("Bot başlatılıyor...")
        application.run_polling()
    except Exception as e:
        logger.error(f"Bot başlatma hatası: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
