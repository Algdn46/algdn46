import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
import nest_asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Async sorunları için
nest_asyncio.apply()

# Logging Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bot.log'),
              logging.StreamHandler()])

# Çevresel Değişkenler
load_dotenv('gateio.env')


# Exchange ve Bot Kurulumu
def initialize_exchange():
    return ccxt.gate({
        'apiKey': os.getenv('GATEIO_API_KEY'),
        'secret': os.getenv('GATEIO_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap'
        },
    })


# Telegram Bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

if not TELEGRAM_TOKEN or not os.getenv('GATEIO_API_KEY') or not os.getenv(
        'GATEIO_SECRET_KEY'):
    raise ValueError("API anahtarları veya Telegram token eksik!")

application = Application.builder().token(TELEGRAM_TOKEN).build()
exchange = initialize_exchange()

# Global Ayarlar
CONFIG = {
    'SYMBOLS': [],
    'running': False,  # Başlangıçta bot çalışmıyor
    'LEVERAGE': 10,
    'RISK_PER_TRADE': 0.02,
    'TIMEFRAMES': ['5m', '15m', '1h'],
    'LSTM_LOOKBACK': 50,
    'EMA_PERIODS': (9, 21),
    'RSI_PERIOD': 14,
    'ATR_PERIOD': 14,
    'chat_ids': set(),
    'signals': {},
    'positions': {},
    'model': None
}


# LSTM Modeli
def create_lstm_model():
    model = Sequential([
        LSTM(64,
             return_sequences=True,
             input_shape=(CONFIG['LSTM_LOOKBACK'], 3)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


CONFIG['model'] = create_lstm_model()


# Veri Yönetimi
async def fetch_ohlcv(symbol: str, timeframe: str, retries=3):
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv,
                              columns=[
                                  'timestamp', 'open', 'high', 'low', 'close',
                                  'volume'
                              ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(
                f"{symbol} {timeframe} veri çekme hatası (deneme {attempt+1}/{retries}): {e}"
            )
            if attempt == retries - 1:
                return None
            await asyncio.sleep(2**attempt)


def calculate_indicators(df: pd.DataFrame):
    try:
        df['EMA9'] = df['close'].ewm(span=CONFIG['EMA_PERIODS'][0],
                                     adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=CONFIG['EMA_PERIODS'][1],
                                      adjust=False).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=CONFIG['RSI_PERIOD']).mean()
        avg_loss = loss.rolling(window=CONFIG['RSI_PERIOD']).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=CONFIG['ATR_PERIOD']).mean()
        return df.dropna()
    except Exception as e:
        logging.error(f"Gösterge hesaplama hatası: {e}")
        return df


# Sinyal Üretimi
async def generate_signal(symbol: str):
    try:
        tf_data = {}
        for timeframe in CONFIG['TIMEFRAMES']:
            df = await fetch_ohlcv(symbol, timeframe)
            if df is not None:
                tf_data[timeframe] = calculate_indicators(df)

        df_5m = tf_data.get('5m')
        df_1h = tf_data.get('1h')
        if df_5m is None or df_1h is None or len(df_5m) < 10 or len(
                df_1h) < 10:
            logging.warning(f"{symbol} için yeterli veri yok.")
            return

        current_price = df_5m['close'].iloc[-1]
        last_ema9 = df_5m['EMA9'].iloc[-2]
        last_ema21 = df_5m['EMA21'].iloc[-2]
        current_ema9 = df_5m['EMA9'].iloc[-1]
        current_ema21 = df_5m['EMA21'].iloc[-1]

        long_signal = (last_ema9 < last_ema21) and (
            current_ema9 > current_ema21) and (df_1h['EMA9'].iloc[-1]
                                               > df_1h['EMA21'].iloc[-1]) and (
                                                   df_5m['RSI'].iloc[-1] < 70)
        short_signal = (last_ema9 > last_ema21) and (
            current_ema9 < current_ema21) and (df_1h['EMA9'].iloc[-1]
                                               < df_1h['EMA21'].iloc[-1]) and (
                                                   df_5m['RSI'].iloc[-1] > 30)

        if not (long_signal or short_signal):
            return

        atr = df_5m['ATR'].iloc[-1]
        entry_price = current_price
        tp = entry_price + (2 * atr) if long_signal else entry_price - (2 *
                                                                        atr)
        sl = entry_price - atr if long_signal else entry_price + atr

        balance = exchange.fetch_balance()['USDT']['free']
        risk_amount = balance * CONFIG['RISK_PER_TRADE']
        position_size = risk_amount / abs(entry_price - sl)

        signal = {
            'symbol': symbol,
            'direction': 'LONG' if long_signal else 'SHORT',
            'entry': entry_price,
            'tp': tp,
            'sl': sl,
            'size': position_size,
            'timestamp': datetime.now().isoformat()
        }

        CONFIG['signals'][symbol] = signal
        logging.info(f"Yeni sinyal üretildi: {signal}")
        await broadcast_signal(signal)
    except Exception as e:
        logging.error(f"{symbol} sinyal üretim hatası: {e}")


# Telegram Entegrasyonu
async def broadcast_signal(signal: dict):
    message = (f"🚨 Yeni Sinyal 🚨\n"
               f"• Sembol: {signal['symbol']}\n"
               f"• Yön: {signal['direction']}\n"
               f"• Giriş: {signal['entry']:.4f}\n"
               f"• TP: {signal['tp']:.4f}\n"
               f"• SL: {signal['sl']:.4f}\n"
               f"• Boyut: {signal['size']:.2f} USDT")

    for chat_id in CONFIG['chat_ids']:
        try:
            await application.bot.send_message(chat_id=chat_id,
                                               text=message,
                                               parse_mode='Markdown')
        except Exception as e:
            logging.error(f"{chat_id} mesaj gönderme hatası: {e}")


# Komutlar
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    CONFIG['chat_ids'].add(chat_id)
    await context.bot.send_message(chat_id=chat_id,
                                   text="✅ Gate.io Trading Bot Aktif\n"
                                   "Sinyaller burada görünecek.\n"
                                   "Komutlar:\n"
                                   "/signals - Sinyal üretimini başlat\n"
                                   "/stop - Sinyal üretimini durdur\n"
                                   "/unsubscribe - Bildirimleri kapat")


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG['running'] = False
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="🛑 Sinyal üretimi durduruldu.")


async def show_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in CONFIG['chat_ids']:
        CONFIG['chat_ids'].add(chat_id)

    if not CONFIG['running']:
        CONFIG['running'] = True
        await context.bot.send_message(
            chat_id=chat_id,
            text=
            "🔄 Sinyal üretimi başlatıldı. Yeni sinyaller burada görünecek.\n"
            "Durdurmak için /stop komutunu kullanın.")
        scheduler = AsyncIOScheduler()
        scheduler.add_job(run_signal_generation, 'interval', seconds=60)
        scheduler.start()
    else:
        if not CONFIG['signals']:
            await context.bot.send_message(
                chat_id=chat_id,
                text="ℹ️ Henüz aktif sinyal bulunmuyor. Lütfen bekleyin...")
            return

        response = ["📊 *Aktif Sinyaller*"]
        for signal in CONFIG['signals'].values():
            response.append(
                f"• {signal['symbol']} - {signal['direction']} "
                f"(Giriş: {signal['entry']:.4f}, TP: {signal['tp']:.4f}, SL: {signal['sl']:.4f})"
            )

        await context.bot.send_message(chat_id=chat_id,
                                       text="\n".join(response),
                                       parse_mode='Markdown')


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in CONFIG['chat_ids']:
        CONFIG['chat_ids'].remove(chat_id)
        await context.bot.send_message(chat_id=chat_id,
                                       text="✅ Sinyal bildirimleri kapatıldı.")
    else:
        await context.bot.send_message(chat_id=chat_id,
                                       text="ℹ️ Zaten bildirim almıyorsunuz.")


# Sinyal Üretim Döngüsü
async def run_signal_generation():
    if not CONFIG['running']:
        return
    for symbol in CONFIG['SYMBOLS'][:20]:  # İlk 20 sembolü kontrol et
        await generate_signal(symbol)


# Ana Döngü
async def main():
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("signals", show_signals))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))

    markets = exchange.load_markets()
    CONFIG['SYMBOLS'] = [
        symbol for symbol, market in markets.items()
        if market['type'] == 'swap' and market['active']
        and market['quote'] == 'USDT'
    ]
    logging.info(f"{len(CONFIG['SYMBOLS'])} adet sembol yüklendi")

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    try:
        await asyncio.Event().wait()  # Botun çalışmasını sürdürmek için
    finally:
        await application.updater.stop()
        await application.stop()


if _name_ == '_main_':
    asyncio.run(main())

    
