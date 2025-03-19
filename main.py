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
from flask import Flask
import threading

# TensorFlow uyarılarını bastırmak için
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
tf.data.experimental.enable_debug_mode()

# Eager execution'ı etkinleştir
tf.config.run_functions_eagerly(True)

# Flask uygulamasını oluştur
app = Flask(__name__)

# Basit bir endpoint ekle
@app.route('/')
def keep_alive():
    return "Bot is alive!", 200

# Flask sunucusunu ayrı bir thread'de çalıştır
def run_flask():
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# Kendi kendine ping atma fonksiyonu
def self_ping():
    while True:
        try:
            service_url = os.getenv("SERVICE_URL", "https://algdn46-bot.onrender.com")
            requests.get(service_url)
            logging.info("Self-ping gönderildi, bot uyanık tutuluyor...")
        except Exception as e:
            logging.error(f"Self-ping hatası: {str(e)}")
        time.sleep(600)

# Özel TLS yapılandırması
context = create_urllib3_context(ssl_minimum_version=ssl.TLSVersion.TLSv1_2)
context.set_ciphers("DEFAULT")
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount("https://", adapter)
session.verify = True

# Binance exchange nesnesini oluştur (spot piyasası için)
exchange = binance({
    "session": session,
    "enableRateLimit": True,
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'options': {'defaultType': 'spot'},
})

# Config ve Log Ayarları
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Sabitler
INTERVAL = '5m'
LOOKBACK = 60
last_signals = {}
last_signal_times = {}
TR_TIMEZONE = timezone(timedelta(hours=3))
scaler = MinMaxScaler(feature_range=(0, 1))
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Yüzde tabanlı SL ve TP oranları
SL_PERCENT = 0.025  # %2.5 aşağıda
TP1_PERCENT = 0.015  # %1.5 yukarıda
TP2_PERCENT = 0.025  # %2.5 yukarıda
TP3_PERCENT = 0.035  # %3.5 yukarıda

# Yalnızca bu gruba sinyal gönderilecek
ALLOWED_GROUP_CHAT_ID = -4652984499  # Senin grubunun chat_id değeri

# Haber takibi için fonksiyon
def fetch_crypto_news():
    try:
        url = "https://cointelegraph.com/tags/bitcoin"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='post-card', limit=5)
        
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
                sentiment_score += 0.2
            elif any(word in news_lower for word in ["negative", "criticize", "bearish", "drop", "fall"]):
                sentiment_score -= 0.2
    return sentiment_score

# Spot piyasasında en fazla yükselen coini bul
def get_top_gainer():
    try:
        tickers = exchange.fetch_tickers()
        spot_tickers = {symbol: ticker for symbol, ticker in tickers.items() if 'USDT' in symbol and '/' in symbol}
        
        sorted_tickers = sorted(
            spot_tickers.items(),
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
        
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model, scaler
    except Exception as e:
        logger.error(f"LSTM modeli eğitme hatası: {str(e)}")
        return None, None

# Fiyatı formatlama fonksiyonu (Güncellendi)
def format_price(price, symbol):
    # Sembole göre basamak sayısını belirle
    if 'BTC' in symbol:
        decimals = 2  # BTC için 2 basamak
    elif 'ETH' in symbol:
        decimals = 2  # ETH için 2 basamak
    elif price < 0.01:
        decimals = 6  # Küçük fiyatlar için 6 basamak
    elif price < 1:
        decimals = 4  # 1 USDT'den küçük fiyatlar için 4 basamak
    else:
        decimals = 3  # Diğer coin'ler için 3 basamak

    # Fiyatı yuvarla ve formatla
    price_str = f"{price:.{decimals}f}"
    # Gereksiz sıfırları kaldır
    if '.' in price_str:
        price_str = price_str.rstrip('0').rstrip('.')
    return price_str

# Fiyatları yuvarlama fonksiyonu (Güncellendi)
def round_price(price, symbol):
    # Sembole göre basamak sayısını belirle
    if 'BTC' in symbol:
        decimals = 2
    elif 'ETH' in symbol:
        decimals = 2
    elif price < 0.01:
        decimals = 6
    elif price < 1:
        decimals = 4
    else:
        decimals = 3
    return round(price, decimals)

async def generate_signal(symbol, model, scaler, news_sentiment):
    try:
        # Anlık fiyatı fetch_ticker ile çek
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        logger.info(f"{symbol} | Anlık fiyat (fetch_ticker): {current_price}")

        # OHLCV verilerini al (trend için)
        ohlcv = exchange.fetch_ohlcv(symbol, INTERVAL, limit=LOOKBACK+5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
        
        # Trend analizi
        last_5_closes = df['close'].tail(5).values
        trend_direction = 'UP' if all(last_5_closes[i] < last_5_closes[i+1] for i in range(len(last_5_closes)-1)) else \
                         'DOWN' if all(last_5_closes[i] > last_5_closes[i+1] for i in range(len(last_5_closes)-1)) else 'NEUTRAL'
        logger.info(f"{symbol} | Trend yönü: {trend_direction}")
        
        # Hacim analizi
        avg_volume = df['volume'].rolling(window=14).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_confirmed = current_volume > avg_volume * 1.5
        logger.info(f"{symbol} | Hacim onayı: {volume_confirmed}, Ortalama hacim: {avg_volume}, Mevcut hacim: {current_volume}")
        
        # LSTM tahmini
        data = df['close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        X_test = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        predicted_price = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        logger.info(f"{symbol} | Tahmin edilen fiyat: {predicted_price}")
        
        # Önceki sinyali kontrol et
        if symbol in last_signals:
            last_signal = last_signals[symbol]
            last_direction, last_entry, last_sl, last_tp = last_signal
            last_tp3 = last_tp[2]
            
            if predicted_price > current_price * 1.002 and last_direction == 'LONG':
                if current_price < last_tp3:
                    logger.info(f"{symbol} | Önceki LONG sinyali nedeniyle sinyal üretilmedi.")
                    return None, None, None, None
            elif predicted_price < current_price * 0.998 and last_direction == 'SHORT':
                if current_price > last_tp3:
                    logger.info(f"{symbol} | Önceki SHORT sinyali nedeniyle sinyal üretilmedi.")
                    return None, None, None, None
        
        # Haber etkisini dahil et
        price_threshold = 0.001
        if news_sentiment > 0:
            price_threshold -= news_sentiment
        elif news_sentiment < 0:
            price_threshold += news_sentiment
        logger.info(f"{symbol} | Fiyat eşiği (haber etkisi dahil): {price_threshold}")
        
        # Sinyal üretimi (yüzde tabanlı SL ve TP)
        if (predicted_price > current_price * (1 + price_threshold) and
            trend_direction in ['UP', 'NEUTRAL']):
            sl = current_price * (1 - SL_PERCENT)
            tp1 = current_price * (1 + TP1_PERCENT)
            tp2 = current_price * (1 + TP2_PERCENT)
            tp3 = current_price * (1 + TP3_PERCENT)
            
            entry = current_price
            sl = round_price(sl, symbol)
            tp1 = round_price(tp1, symbol)
            tp2 = round_price(tp2, symbol)
            tp3 = round_price(tp3, symbol)
            
            logger.info(f"{symbol} | LONG sinyali üretildi - Giriş: {entry}, SL: {sl}, TP1: {tp1}, TP2: {tp2}, TP3: {tp3}")
            return 'LONG', entry, sl, (tp1, tp2, tp3)
        elif (predicted_price < current_price * (1 - price_threshold) and
              trend_direction in ['DOWN', 'NEUTRAL']):
            sl = current_price * (1 + SL_PERCENT)
            tp1 = current_price * (1 - TP1_PERCENT)
            tp2 = current_price * (1 - TP2_PERCENT)
            tp3 = current_price * (1 - TP3_PERCENT)
            
            entry = current_price
            sl = round_price(sl, symbol)
            tp1 = round_price(tp1, symbol)
            tp2 = round_price(tp2, symbol)
            tp3 = round_price(tp3, symbol)
            
            logger.info(f"{symbol} | SHORT sinyali üretildi - Giriş: {entry}, SL: {sl}, TP1: {tp1}, TP2: {tp2}, TP3: {tp3}")
            return 'SHORT', entry, sl, (tp1, tp2, tp3)
        
        logger.info(f"{symbol} | Sinyal üretilemedi.")
        return None, None, None, None
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}")
        return None, None, None, None

async def format_telegram_message(symbol, direction, entry, sl, tp):
    try:
        clean_symbol = symbol.split('/')[0] + '/USDT'
        direction_text = '🚀 Long' if direction == 'LONG' else '🔻 Short'
        tp1, tp2, tp3 = tp
        
        # Fiyatları formatla
        entry_formatted = format_price(entry, symbol)
        sl_formatted = format_price(sl, symbol)
        tp1_formatted = format_price(tp1, symbol)
        tp2_formatted = format_price(tp2, symbol)
        tp3_formatted = format_price(tp3, symbol)
        
        message = f"""
🚦✈️ {clean_symbol} {direction_text}
━━━━━━━━━━━━━━
🪂 Giriş: {entry_formatted}
🚫 SL: {sl_formatted}
🎯 TP1: {tp1_formatted}
🎯 TP2: {tp2_formatted}
🎯 TP3: {tp3_formatted}
🕒 Zaman: {datetime.now(TR_TIMEZONE).strftime('%H:%M')}
"""
        return message
    except Exception as e:
        logger.error(f"Mesaj formatlama hatası: {str(e)}")
        return "Mesaj formatlama hatası oluştu!"

async def scan_symbols(context: ContextTypes.DEFAULT_TYPE, models: dict, scalers: dict):
    try:
        logger.info("Sinyaller taranıyor...")
        for attempt in range(3):
            try:
                markets = exchange.load_markets()
                logger.info("Markets başarıyla yüklendi.")
                break
            except Exception as e:
                logger.error(f"load_markets attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    await context.bot.send_message(chat_id=ALLOWED_GROUP_CHAT_ID, text="Binance verileri yüklenemedi, tekrar dene!")
                    return
        
        # Haberleri çek ve analiz et
        news_data = fetch_crypto_news()
        news_sentiment = analyze_news_sentiment(news_data)
        logger.info(f"Haber duygu analizi skoru: {news_sentiment}")
        
        # En fazla yükselen coini bul
        top_gainer = get_top_gainer()
        if top_gainer:
            logger.info(f"En fazla yükselen coin: {top_gainer}")
        
        symbols = [s for s in markets if markets[s]['type'] == 'spot' and markets[s]['active'] and 'USDT' in s]
        logger.info(f"Taranacak sembol sayısı: {len(symbols)}")
        
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
                    # Yalnızca ALLOWED_GROUP_CHAT_ID'ye sinyal gönder
                    await context.bot.send_message(
                        chat_id=ALLOWED_GROUP_CHAT_ID,
                        text=message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sinyal gönderildi (chat_id: {ALLOWED_GROUP_CHAT_ID}): {message}")
                    last_signals[top_gainer] = current_signal
                    last_signal_times[top_gainer] = datetime.now(TR_TIMEZONE)
                    found_signal = True
                    time.sleep(1)
            except Exception as e:
                logger.error(f"{top_gainer} tarama hatası: {str(e)}")
            symbols.remove(top_gainer)
        
        # Diğer sembolleri tara
        for symbol in symbols:
            try:
                if symbol in last_signal_times:
                    last_time = last_signal_times[symbol]
                    if (datetime.now(TR_TIMEZONE) - last_time).total_seconds() < 300:
                        logger.info(f"{symbol} | Son 5 dakika içinde sinyal üretildi, atlanıyor.")
                        continue
                
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
                    # Yalnızca ALLOWED_GROUP_CHAT_ID'ye sinyal gönder
                    await context.bot.send_message(
                        chat_id=ALLOWED_GROUP_CHAT_ID,
                        text=message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sinyal gönderildi (chat_id: {ALLOWED_GROUP_CHAT_ID}): {message}")
                    last_signals[symbol] = current_signal
                    last_signal_times[symbol] = datetime.now(TR_TIMEZONE)
                    found_signal = True
                    time.sleep(1)
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {str(e)}")
        
        if not found_signal:
            await context.bot.send_message(chat_id=ALLOWED_GROUP_CHAT_ID, text="Sinyal bulunamadı ede. Az sabret.")
    except Exception as e:
        logger.error(f"Genel tarama hatası: {str(e)}")
        await context.bot.send_message(chat_id=ALLOWED_GROUP_CHAT_ID, text="Bir hata oluştu, tekrar dene!")

async def continuous_scan(context: ContextTypes.DEFAULT_TYPE):
    models = context.bot_data.get('models', {})
    scalers = context.bot_data.get('scalers', {})
    while True:
        try:
            logger.info("Sürekli sinyal tarama başlıyor...")
            markets = exchange.load_markets()
            logger.info("Markets başarıyla yüklendi (continuous_scan).")
            
            # Haberleri çek ve analiz et
            news_data = fetch_crypto_news()
            news_sentiment = analyze_news_sentiment(news_data)
            logger.info(f"Haber duygu analizi skoru: {news_sentiment}")
            
            # En fazla yükselen coini bul
            top_gainer = get_top_gainer()
            if top_gainer:
                logger.info(f"En fazla yükselen coin: {top_gainer}")
            
            symbols = [s for s in markets if markets[s]['type'] == 'spot' and markets[s]['active'] and 'USDT' in s]
            logger.info(f"Taranacak sembol sayısı (continuous_scan): {len(symbols)}")
            
            # Önce en fazla yükselen coini tara
            found_signal = False
            if top_gainer in symbols:
                try:
                    if top_gainer in last_signal_times:
                        last_time = last_signal_times[top_gainer]
                        if (datetime.now(TR_TIMEZONE) - last_time).total_seconds() < 300:
                            logger.info(f"{top_gainer} | Son 5 dakika içinde sinyal üretildi, atlanıyor.")
                            symbols.remove(top_gainer)
                            continue
                    
                    if top_gainer not in models or random.random() < 0.1:
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
                        # Yalnızca ALLOWED_GROUP_CHAT_ID'ye sinyal gönder
                        await context.bot.send_message(
                            chat_id=ALLOWED_GROUP_CHAT_ID,
                            text=message,
                            parse_mode='HTML'
                        )
                        logger.info(f"Sinyal gönderildi (chat_id: {ALLOWED_GROUP_CHAT_ID}): {message}")
                        last_signals[top_gainer] = current_signal
                        last_signal_times[top_gainer] = datetime.now(TR_TIMEZONE)
                        found_signal = True
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"{top_gainer} tarama hatası: {str(e)}")
                symbols.remove(top_gainer)
            
            # Diğer sembolleri tara
            for symbol in symbols:
                if symbol in last_signal_times:
                    last_time = last_signal_times[symbol]
                    if (datetime.now(TR_TIMEZONE) - last_time).total_seconds() < 300:
                        logger.info(f"{symbol} | Son 5 dakika içinde sinyal üretildi, atlanıyor.")
                        continue
                
                if symbol not in models or random.random() < 0.1:
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
                    # Yalnızca ALLOWED_GROUP_CHAT_ID'ye sinyal gönder
                    await context.bot.send_message(
                        chat_id=ALLOWED_GROUP_CHAT_ID,
                        text=message,
                        parse_mode='HTML'
                    )
                    logger.info(f"Sinyal gönderildi (chat_id: {ALLOWED_GROUP_CHAT_ID}): {message}")
                    last_signals[symbol] = current_signal
                    last_signal_times[symbol] = datetime.now(TR_TIMEZONE)
                    found_signal = True
                    time.sleep(1)
            if not found_signal:
                logger.info("Sinyal bulunamadı, 300 saniye bekleniyor...")
                await context.bot.send_message(chat_id=ALLOWED_GROUP_CHAT_ID, text="Sinyal bulunamadı ede. Az sabret.")
            context.bot_data['models'] = models
            context.bot_data['scalers'] = scalers
            await asyncio.sleep(300)
        except Exception as e:
            logger.error(f"Sürekli tarama hatası: {str(e)}")
            await context.bot.send_message(chat_id=ALLOWED_GROUP_CHAT_ID, text="Bir hata oluştu, tekrar dene!")
            await asyncio.sleep(300)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    logger.info(f"Start komutu alındı, chat_id: {chat_id}, chat_type: {chat_type}")
    
    # Eğer komut ALLOWED_GROUP_CHAT_ID'den gelmiyorsa, yalnızca bir uyarı mesajı gönder
    if chat_id != ALLOWED_GROUP_CHAT_ID:
        await update.message.reply_text("Bu bot yalnızca belirli bir gruba sinyal gönderir. Lütfen gruba katılın.")
        return
    
    # Gruba hoş geldin mesajı gönder
    await update.message.reply_text("🚀 Kemerini tak dostum, sinyaller geliyor...")
    
    # İlk taramayı başlat
    await scan_symbols(context, context.bot_data.get('models', {}), context.bot_data.get('scalers', {}))
    
    # Sürekli taramayı başlat (yalnızca bir kez başlatılması için kontrol et)
    if not context.job_queue.get_jobs_by_name("continuous_scan"):
        context.job_queue.run_repeating(continuous_scan, interval=300, first=5, name="continuous_scan")
        logger.info("Sürekli tarama başlatıldı.")

def main():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    ping_thread = threading.Thread(target=self_ping)
    ping_thread.daemon = True
    ping_thread.start()

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
         port = int(os.getenv("PORT", 8000))  # Render PORT ortam değişkenini kullan, yoksa 8000
         app.run(host="0.0.0.0", port=port)
