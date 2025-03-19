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
from bs4 import BeautifulSoup
import ssl
from urllib3.util.ssl_ import create_urllib3_context
from ccxt import binance
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import pickle
import random

# TensorFlow uyarılarını bastırmak için
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.data.experimental.enable_debug_mode()

# Eager execution'ı etkinleştir
tf.config.run_functions_eagerly(True)

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
RISK_RATIO = 1.0
LOOKBACK = 60
last_signals = {}
TR_TIMEZONE = timezone(timedelta(hours=3))
scaler = MinMaxScaler(feature_range=(0, 1))
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Haber takibi için fonksiyon
def fetch_crypto_news():
    try:
        url = "https://cointelegraph.com/tags/bitcoin"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='post-card', limit=5)  # Son 5 haberi al
        
        news_data = []
        for article in articles:
            title = article.find('span', class_='post-card__title').text.strip()
            news_data.append(title)
        return news_data
    except Exception as e:
        logger.error(f"Haber çekme hatası: {str(e)}")
        return []

# Duygu analizi (basit kural tabanlı)
def analyze_news_sentiment(news_data):
    sentiment_score = 0
    for news in news_data:
        news_lower = news.lower()
        if "elon musk" in news_lower and "bitcoin" in news_lower:
            if any(word in news_lower for word in ["positive", "support", "bullish", "soar", "rise"]):
                sentiment_score += 0.2  # Olumlu haber, %20 artış etkisi
            elif any(word in news_lower for word in ["negative", "criticize", "bearish", "drop", "fall"]):
                sentiment_score -= 0.2  # Olumsuz haber, %20 düşüş etkisi
    return sentiment_score

# Futures piyasasında en fazla yükselen coini bul
def get_top_gainer():
    try:
        tickers = exchange.fetch_tickers()
        futures_tickers = {symbol: ticker for symbol, ticker in tickers.items() if 'USDT' in symbol and ticker.get('info', {}).get('contractType') == 'PERPETUAL'}
        
        # 24 saatlik değişim oranına göre sırala
        sorted_tickers = sorted(
            futures_tickers.items(),
            key=lambda x: x[1].get('percentage', 0),
            reverse=True
        )
        if sorted_tickers:
            top_symbol = sorted_tickers[0][0]
            return top_symbol
        return None
    except Exception as e:
        logger.error(f"Top gainer bulma hatası: {str(e)}")
        return None

# LSTM Modelini oluştur
def create_lstm_model():
    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Modeli eğit ve güncelle
def train_lstm_model(symbol, retrain=False):
    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_lstm_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol.replace('/', '_')}_scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and not retrain:
        logger.info(f"{symbol} için model ve scaler yükleniyor...")
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, INTERVAL, limit=10000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
        
        data = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X_train, y_train = [], []
        for i in range(LOOKBACK, len(scaled_data)):
            X_train.append(scaled_data[i-LOOKBACK:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        model = create_lstm_model()
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
        
        # Modeli ve scaler'ı kaydet
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model, scaler
    except Exception as e:
        logger.error(f"LSTM modeli eğitme hatası: {str(e)}")
        return None, None

# Fiyatları yuvarlama fonksiyonu
def round_price(price, symbol):
    if 'BTC' in symbol:
        return round(price, 3)
    elif 'ETH' in symbol:
        return round(price, 1)
    else:
        return round(price, 3)

async def generate_signal(symbol, model, scaler, news_sentiment):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, INTERVAL, limit=LOOKBACK+5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
        
        # Trend analizi
        last_5_closes = df['close'].tail(5).values
        trend_direction = 'UP' if all(last_5_closes[i] < last_5_closes[i+1] for i in range(len(last_5_closes)-1)) else \
                         'DOWN' if all(last_5_closes[i] > last_5_closes[i+1] for i in range(len(last_5_closes)-1)) else 'NEUTRAL'
        
        # Hacim analizi
        avg_volume = df['volume'].rolling(window=14).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_confirmed = current_volume > avg_volume * 1.5
        
        # Volatilite analizi
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = ranges.rolling(window=14).mean().iloc[-1]
        volatility_confirmed = atr > df['close'].iloc[-1] * 0.005
        
        # LSTM tahmini
        data = df['close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        X_test = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        predicted_price = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        
        last = df.iloc[-1]
        current_price = last['close']
        
        # Önceki sinyali kontrol et
        if symbol in last_signals:
            last_signal = last_signals[symbol]
            last_direction, last_entry, last_sl, last_tp = last_signal
            last_tp3 = last_tp[2]
            
            if predicted_price > current_price * 1.002 and last_direction == 'LONG':
                if current_price < last_tp3:
                    return None, None, None, None
            elif predicted_price < current_price * 0.998 and last_direction == 'SHORT':
                if current_price > last_tp3:
                    return None, None, None, None
        
        # Haber etkisini dahil et
        price_threshold = 0.002  # Varsayılan eşik %0.2
        # Sinyal üretimini kolaylaştırmak için (opsiyonel):
        # price_threshold = 0.001  # Varsayılan eşik %0.1
        
        if news_sentiment > 0:
            price_threshold -= news_sentiment  # Olumlu haber, eşiği düşür
        elif news_sentiment < 0:
            price_threshold += news_sentiment  # Olumsuz haber, eşiği artır
        
        # Sinyal üretimi
        if (predicted_price > current_price * (1 + price_threshold) and
            trend_direction in ['UP', 'NEUTRAL'] and
            volume_confirmed and volatility_confirmed):
            # Sinyal üretimini kolaylaştırmak için (opsiyonel):
            # if (predicted_price > current_price * (1 + price_threshold) and
            #     trend_direction in ['UP', 'NEUTRAL']):
            
            sl = last['low'] - (atr * RISK_RATIO * 2.0)
            tp1 = last['close'] + (atr * RISK_RATIO * 2.0)
            tp2 = last['close'] + (atr * RISK_RATIO * 3.0)
            tp3 = last['close'] + (atr * RISK_RATIO * 4.0)
            
            entry = round_price(last['close'], symbol)
            sl = round_price(sl, symbol)
            tp1 = round_price(tp1, symbol)
            tp2 = round_price(tp2, symbol)
            tp3 = round_price(tp3, symbol)
            
            return 'LONG', entry, sl, (tp1, tp2, tp3)
        elif (predicted_price < current_price * (1 - price_threshold) and
              trend_direction in ['DOWN', 'NEUTRAL'] and
              volume_confirmed and volatility_confirmed):
            # Sinyal üretimini kolaylaştırmak için (opsiyonel):
            # elif (predicted_price < current_price * (1 - price_threshold) and
            #       trend_direction in ['DOWN', 'NEUTRAL']):
            
            sl = last['high'] + (atr * RISK_RATIO * 2.0)
            tp1 = last['close'] - (atr * RISK_RATIO * 2.0)
            tp2 = last['close'] - (atr * RISK_RATIO * 3.0)
            tp3 = last['close'] - (atr * RISK_RATIO * 4.0)
            
            entry = round_price(last['close'], symbol)
            sl = round_price(sl, symbol)
            tp1 = round_price(tp1, symbol)
            tp2 = round_price(tp2, symbol)
            tp3 = round_price(tp3, symbol)
            
            return 'SHORT', entry, sl, (tp1, tp2, tp3)
        
        return None, None, None, None
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}")
        return None, None, None, None

async def format_telegram_message(symbol, direction, entry, sl, tp):
    try:
        clean_symbol = symbol.replace(':USDT-', '/USDT').split('/')[0] + '/USDT'
        direction_text = '🚀 Long' if direction == 'LONG' else '🔻 Short'
        tp1, tp2, tp3 = tp
        message = f"""
🚦✈️ {clean_symbol} {direction_text}
━━━━━━━━━━━━━━
🪂 Giriş: {entry}
🚫 SL: {sl}
🎯 TP1: {tp1}
🎯 TP2: {tp2}
🎯 TP3: {tp3}
🕒 Zaman: {datetime.now(TR_TIMEZONE).strftime('%H:%M')}
"""
        return message
    except Exception as e:
        logger.error(f"Mesaj formatlama hatası: {str(e)}")
        return "Mesaj formatlama hatası oluştu!"

async def scan_symbols(context: ContextTypes.DEFAULT_TYPE, chat_id: int, models: dict, scalers: dict):
    try:
        logger.info("Sinyaller taranıyor...")
        for attempt in range(3):
            try:
                markets = exchange.load_markets()
                break
            except Exception as e:
                logger.error(f"load_markets attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    await context.bot.send_message(chat_id=chat_id, text="Binance verileri yüklenemedi, tekrar dene!")
                    return
        
        # Haberleri çek ve analiz et
        news_data = fetch_crypto_news()
        news_sentiment = analyze_news_sentiment(news_data)
        logger.info(f"Haber duygu analizi skoru: {news_sentiment}")
        
        # En fazla yükselen coini bul
        top_gainer = get_top_gainer()
        if top_gainer:
            logger.info(f"En fazla yükselen coin: {top_gainer}")
        
        symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
        
        # Önce en fazla yükselen coini tara
        found_signal = False
        if top_gainer in symbols:
            try:
                if top_gainer not in models:
                    model, scaler = train_lstm_model(top_gainer)
                    if model is None or scaler is None:
                        symbols.remove(top_gainer)
                    else:
                        models[top_gainer] = model
                        scalers[top_gainer] = scaler
                model = models[top_gainer]
                scaler = scalers[top_gainer]
                direction, entry, sl, tp = await generate_signal(top_gainer, model, scaler, news_sentiment)
                if direction and entry:
                    current_signal = (direction, entry, sl, tp)
                    message = await format_telegram_message(top_gainer, direction, entry, sl, tp)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sinyal gönderildi: {message}")
                    last_signals[top_gainer] = current_signal
                    found_signal = True
                    time.sleep(1)
            except Exception as e:
                logger.error(f"{top_gainer} tarama hatası: {str(e)}")
            symbols.remove(top_gainer)  # Top gainer'ı listeden çıkar
        
        # Diğer sembolleri tara
        for symbol in symbols:
            try:
                if symbol not in models:
                    model, scaler = train_lstm_model(symbol)
                    if model is None or scaler is None:
                        continue
                    models[symbol] = model
                    scalers[symbol] = scaler
                model = models[symbol]
                scaler = scalers[symbol]
                direction, entry, sl, tp = await generate_signal(symbol, model, scaler, news_sentiment)
                if direction and entry:
                    current_signal = (direction, entry, sl, tp)
                    message = await format_telegram_message(symbol, direction, entry, sl, tp)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sinyal gönderildi: {message}")
                    last_signals[symbol] = current_signal
                    found_signal = True
                    time.sleep(1)
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {str(e)}")
        
        if not found_signal:
            await context.bot.send_message(chat_id=chat_id, text="Sinyal bulunamadı ede. Az sabret.")
    except Exception as e:
        logger.error(f"Genel tarama hatası: {str(e)}")
        await context.bot.send_message(chat_id=chat_id, text="Bir hata oluştu, tekrar dene!")

async def continuous_scan(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.bot_data.get('chat_id')
    models = context.bot_data.get('models', {})
    scalers = context.bot_data.get('scalers', {})
    while True:
        try:
            logger.info("Sürekli sinyal tarama başlıyor...")
            markets = exchange.load_markets()
            
            # Haberleri çek ve analiz et
            news_data = fetch_crypto_news()
            news_sentiment = analyze_news_sentiment(news_data)
            logger.info(f"Haber duygu analizi skoru: {news_sentiment}")
            
            # En fazla yükselen coini bul
            top_gainer = get_top_gainer()
            if top_gainer:
                logger.info(f"En fazla yükselen coin: {top_gainer}")
            
            symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
            
            # Önce en fazla yükselen coini tara
            found_signal = False
            if top_gainer in symbols:
                try:
                    if top_gainer not in models or random.random() < 0.1:  # %10 ihtimalle modeli güncelle
                        model, scaler = train_lstm_model(top_gainer, retrain=True)
                        if model is None or scaler is None:
                            symbols.remove(top_gainer)
                            continue
                        models[top_gainer] = model
                        scalers[top_gainer] = scaler
                    model = models[top_gainer]
                    scaler = scalers[top_gainer]
                    direction, entry, sl, tp = await generate_signal(top_gainer, model, scaler, news_sentiment)
                    if direction and entry:
                        current_signal = (direction, entry, sl, tp)
                        message = await format_telegram_message(top_gainer, direction, entry, sl, tp)
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML'
                        )
                        logger.info(f"Sinyal gönderildi: {message}")
                        last_signals[top_gainer] = current_signal
                        found_signal = True
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"{top_gainer} tarama hatası: {str(e)}")
                symbols.remove(top_gainer)
            
            # Diğer sembolleri tara
            for symbol in symbols:
                if symbol not in models or random.random() < 0.1:  # %10 ihtimalle modeli güncelle
                    model, scaler = train_lstm_model(symbol, retrain=True)
                    if model is None or scaler is None:
                        continue
                    models[symbol] = model
                    scalers[symbol] = scaler
                model = models[symbol]
                scaler = scalers[symbol]
                direction, entry, sl, tp = await generate_signal(symbol, model, scaler, news_sentiment)
                if direction and entry:
                    current_signal = (direction, entry, sl, tp)
                    message = await format_telegram_message(symbol, direction, entry, sl, tp)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sinyal gönderildi: {message}")
                    last_signals[symbol] = current_signal
                    found_signal = True
                    time.sleep(1)
            if not found_signal:
                logger.info("Sinyal bulunamadı, 300 saniye bekleniyor...")
            context.bot_data['models'] = models
            context.bot_data['scalers'] = scalers
            await asyncio.sleep(300)  # Tarama aralığını 300 saniyeye çıkardık
        except Exception as e:
            logger.error(f"Sürekli tarama hatası: {str(e)}")
            await asyncio.sleep(300)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    context.bot_data['chat_id'] = chat_id
    context.bot_data['models'] = {}
    context.bot_data['scalers'] = {}
    await update.message.reply_text("🚀 Kemerini tak dostum, sinyaller geliyor...")
    await scan_symbols(context, chat_id, context.bot_data['models'], context.bot_data['scalers'])
    context.job_queue.run_repeating(continuous_scan, interval=300, first=5)  # Tarama aralığını 300 saniyeye çıkardık

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
        logger.error(f"Bot başlatma hatası: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
