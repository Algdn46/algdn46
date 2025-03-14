import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime
from telegram.ext import Application, CommandHandler
from dotenv import load_dotenv

# Config ve Log Ayarları
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Binance Futures API Konfigürasyonu
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# Global Sabitler
INTERVAL = '5m'

async def calculate_technical_indicators(df):
    """Basit EMA hesaplamaları ile teknik göstergeler"""
    try:
        closes = df['close'].values
        
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = ranges.rolling(window=14).mean()
        
        return df.dropna()
    except Exception as e:
        logger.error(f"Gösterge hesaplama hatası: {str(e)}")
        return df

async def generate_signal(symbol):
    """Gelişmiş sinyal üretme mantığı"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, INTERVAL, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = await calculate_technical_indicators(df)
        
        if df.empty:
            return None, None, None, None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ema_cross = (prev['EMA9'] < prev['EMA21']) and (last['EMA9'] > last['EMA21'])
        rsi_ok = last['RSI'] < 65
        
        if ema_cross and rsi_ok:
            sl = last['low'] - (last['ATR'] * 1.5)
            tp = last['close'] + (last['ATR'] * 3)
            return 'LONG', last['close'], sl, tp

        ema_death_cross = (prev['EMA9'] > prev['EMA21']) and (last['EMA9'] < last['EMA21'])
        rsi_overbought = last['RSI'] > 35
        
        if ema_death_cross and rsi_overbought:
            sl = last['high'] + (last['ATR'] * 1.5)
            tp = last['close'] - (last['ATR'] * 3)
            return 'SHORT', last['close'], sl, tp

        return None, None, None, None
        
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}")
        return None, None, None, None

async def format_telegram_message(symbol, direction, entry, sl, tp):
    """Telegram mesaj formatlama"""
    try:
        clean_symbol = symbol.replace(':USDT', '').replace(':BTC', '').replace(':ETH', '').replace(':BUSD', '')
        return f"""
📈 {clean_symbol} <span style="color: {'#00ff00' if direction == 'LONG' else '#ff0000'}"><b>{direction}</b></span>
━━━━━━━━━━━━━━
🎯 Giriş: {entry:,.3f}".replace(".000", "")
🛑 SL : {sl:,.3f}".replace(".000", "")
🎯 TP : {tp:,.3f}".replace(".000", "")
"""
    except Exception as e:
        logger.error(f"Mesaj formatlama hatası: {str(e)}")
        return ""

async def scan_symbols(update, context):
    """Sembol tarama ve sinyal gönderme"""
    try:
        await update.message.reply_text("Sinyaller taranıyor, biraz bekle... 🚀")
        markets = exchange.load_markets()
        symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
        
        found_signal = False
        for symbol in symbols:
            try:
                direction, entry, sl, tp = await generate_signal(symbol)
                if direction and entry:
                    message = await format_telegram_message(symbol, direction, entry, sl, tp)
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=message,
                        parse_mode='HTML'
                    )
                    found_signal = True
                    time.sleep(1)
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {str(e)}")
        
        if not found_signal:
            await update.message.reply_text("Şu anda geçerli sinyal bulunamadı. Daha sonra tekrar dene!")
    except Exception as e:
        logger.error(f"Genel tarama hatası: {str(e)}")
        await update.message.reply_text("Bir hata oluştu, tekrar dene!")

async def start(update, context):
    """Start komutuyla sinyal taramayı başlat"""
    await update.message.reply_text("Woow! 🚀 Kemerini tak dostum, sinyaller geliyor...")
    await scan_symbols(update, context)

if __name__ == '__main__':
    # .env dosyasını yükle
    load_dotenv("config.env")

    # Token'ı ortam değişkeninden al
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN bulunamadı!")
        exit(1)

    # Botu başlat
    try:
        application = Application.builder().token(token).build()
        application.add_handler(CommandHandler("start", start))
        application.run_polling()
    except Exception as e:
        logger.error(f"Bot başlatma hatası: {str(e)}")
