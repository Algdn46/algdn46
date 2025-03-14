import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
import talib
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from dotenv import load_dotenv

# **Konfigürasyon ve Log Ayarları**
load_dotenv('config.env')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# **Binance Futures API Konfigürasyonu**
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# **Global Değişkenler**
INTERVAL = '5m'
CHECK_INTERVAL = 300  # 5 dakika
LEVERAGE = int(os.getenv('LEVERAGE', 10))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))

# **Telegram Bot Konfigürasyonu**
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
application = Application.builder().token(TELEGRAM_TOKEN).build()

# **Marketleri Başlangıçta Yükle**
try:
    markets = exchange.load_markets()
except Exception as e:
    logger.error(f"Market verisi yüklenirken hata: {str(e)}")
    markets = {}

def fetch_ohlcv_with_retry(symbol, interval, limit=100, max_retries=3):
    """Binance'den OHLCV verisini çekme (hata yönetimi ile)"""
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        except ccxt.RateLimitExceeded:
            logger.warning(f"{symbol} | Rate limit aşıldı, tekrar deneniyor ({attempt + 1}/{max_retries})...")
            time.sleep(exchange.rateLimit / 1000.0)  # Milisaniye cinsinden rate limit
        except Exception as e:
            logger.error(f"{symbol} | OHLCV çekme hatası: {str(e)}")
    return None

def calculate_ema(closes, timeperiod):
    """EMA hesaplama"""
    if len(closes) < timeperiod:
        return None
    return talib.EMA(closes, timeperiod=timeperiod)

def calculate_rsi(closes, timeperiod):
    """RSI hesaplama"""
    if len(closes) < timeperiod:
        return None
    return talib.RSI(closes, timeperiod=timeperiod)

def calculate_macd(closes, fastperiod, slowperiod, signalperiod):
    """MACD hesaplama"""
    if len(closes) < slowperiod:
        return None, None
    macd, signal, _ = talib.MACD(closes, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return macd, signal

def calculate_atr(highs, lows, closes, timeperiod):
    """ATR hesaplama"""
    if len(highs) < timeperiod:
        return None
    return talib.ATR(highs, lows, closes, timeperiod=timeperiod)

def calculate_technical_indicators(df):
    """TA-Lib ile teknik göstergeleri hesaplar"""
    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        df['EMA9'] = calculate_ema(closes, 9)
        df['EMA21'] = calculate_ema(closes, 21)
        df['RSI'] = calculate_rsi(closes, 14)
        
        macd, signal = calculate_macd(closes, 12, 26, 9)
        df['MACD'] = macd
        df['Signal'] = signal
        
        df['ATR'] = calculate_atr(highs, lows, closes, 14)
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Gösterge hesaplama hatası: {str(e)}")
        return df

def generate_signal(symbol, interval, limit=100):
    """Sinyal üretme fonksiyonu"""
    try:
        ohlcv = fetch_ohlcv_with_retry(symbol, interval, limit)
        if ohlcv is None:
            return None, None, None, None
        
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = calculate_technical_indicators(df)
        
        if df.empty:
            return None, None, None, None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ema_cross = (prev['EMA9'] < prev['EMA21']) and (last['EMA9'] > last['EMA21'])
        macd_bullish = last['MACD'] > last['Signal']
        rsi_ok = last['RSI'] < 65
        
        if ema_cross and macd_bullish and rsi_ok:
            sl = last['low'] - (last['ATR'] * 1.5)
            tp = last['close'] + (last['ATR'] * 3)
            return 'LONG', last['close'], sl, tp

        ema_death_cross = (prev['EMA9'] > prev['EMA21']) and (last['EMA9'] < last['EMA21'])
        macd_bearish = last['MACD'] < last['Signal']
        rsi_overbought = last['RSI'] > 35
        
        if ema_death_cross and macd_bearish and rsi_overbought:
            sl = last['high'] + (last['ATR'] * 1.5)
            tp = last['close'] - (last['ATR'] * 3)
            return 'SHORT', last['close'], sl, tp

        return None, None, None, None
        
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}")
        return None, None, None, None

def format_telegram_message(symbol, direction, entry, sl, tp):
    """Telegram mesaj formatlama"""
    try:
        clean_symbol = symbol.replace(':USDT', '').replace(':BTC', '').replace(':ETH', '').replace(':BUSD', '')
        return f"""
<b>{clean_symbol} {direction}</b>
━━━━━━━━━━━━━━
 Giriş: {entry:,.3f}
 SL : {sl:,.3f}
 TP : {tp:,.3f}
"""
    except Exception as e:
        logger.error(f"Mesaj formatlama hatası: {str(e)}")
        return ""

async def scan_symbols(context: CallbackContext):
    """Sembol tarama ve sinyal gönderme"""
    try:
        symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
        
        for symbol in symbols:
            try:
                direction, entry, sl, tp = generate_signal(symbol, INTERVAL)
                if direction and entry:
                    message = format_telegram_message(symbol, direction, entry, sl, tp)
                    await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='HTML')
                    time.sleep(1)  # Rate limit için
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {str(e)}")
    except Exception as e:
        logger.error(f"Sembol tarama hatası: {str(e)}")

# **Ana Çalıştırıcı**
if __name__ == "__main__":
    application.job_queue.run_repeating(scan_symbols, interval=CHECK_INTERVAL)
    application.run_polling()
