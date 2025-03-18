import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
import requests
import ssl
from urllib3.util.ssl_ import create_urllib3_context
from ccxt import binance

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
RISK_RATIO = 0.2  # Risk oranını %0.2'ye yükselttim

# Son sinyalleri takip etmek için bir sözlük
last_signals = {}

async def calculate_technical_indicators(df):
    try:
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
        logger.error(f"Gösterge hesaplama hatası: {str(e)}", exc_info=True)
        return df

async def generate_signal(symbol):
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
        rsi_ok = last['RSI'] < 70  # RSI koşulunu gevşettik
        
        if ema_cross and rsi_ok:
            sl = last['low'] - (last['ATR'] * RISK_RATIO)
            tp = last['close'] + (last['ATR'] * RISK_RATIO * 1.5)
            return 'LONG', last['close'], sl, tp

        ema_death_cross = (prev['EMA9'] > prev['EMA21']) and (last['EMA9'] < last['EMA21'])
        rsi_overbought = last['RSI'] > 30  # RSI koşulunu gevşettik
        
        if ema_death_cross and rsi_overbought:
            sl = last['high'] + (last['ATR'] * RISK_RATIO)
            tp = last['close'] - (last['ATR'] * RISK_RATIO * 1.5)
            return 'SHORT', last['close'], sl, tp

        return None, None, None, None
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}", exc_info=True)
        return None, None, None, None

async def format_telegram_message(symbol, direction, entry, sl, tp):
    try:
        # Sembolü sadeleştir (USDT ile biten format)
        clean_symbol = symbol.replace(':USDT-', '/USDT').split('/')[0] + '/USDT'
        color = '<span style="color:green">Long</span>' if direction == 'LONG' else '<span style="color:red">Short</span>'
        message = f"""
🚦✈️ {clean_symbol} {color}
━━━━━━━━━━━━━━
🪂 Giriş: {entry:.3f}
🚫 SL: {sl:.3f}
🎯 TP: {tp:.3f}
"""
        return message
    except Exception as e:
        logger.error(f"Mesaj formatlama hatası: {str(e)}", exc_info=True)
        return "Mesaj formatlama hatası oluştu!"

async def scan_symbols(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
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
                direction, entry, sl, tp = await generate_signal(symbol)
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
    while True:
        try:
            logger.info("Sürekli sinyal tarama başlıyor...")
            markets = exchange.load_markets()
            symbols = [s for s in markets if markets[s]['type'] == 'future' and markets[s]['active']]
            found_signal = False
            for symbol in symbols:
                direction, entry, sl, tp = await generate_signal(symbol)
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
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Sürekli tarama hatası: {str(e)}", exc_info=True)
            await asyncio.sleep(60)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    context.bot_data['chat_id'] = chat_id
    await update.message.reply_text("🚀 Kemerini tak dostum, sinyaller geliyor...")
    await scan_symbols(context, chat_id)
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
