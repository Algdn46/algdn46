# Gerekli Kütüphaneler
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

# Async sorunları için
nest_asyncio.apply()

# Logging Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
)

# Çevresel Değişkenler
load_dotenv('gateio.env')

# 1. Exchange ve Bot Kurulumu
def initialize_exchange():
    return ccxt.gate({
        'apiKey': os.getenv('GATEIO_API_KEY'),
        'secret': os.getenv('GATEIO_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

from telegram.ext import Application

# Telegram Bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN or not os.getenv('GATEIO_API_KEY') or not os.getenv('GATEIO_SECRET_KEY'):
    raise ValueError("API anahtarları veya Telegram token eksik!")

# Application nesnesi oluştur
application = Application.builder().token(TELEGRAM_TOKEN).build()
exchange = initialize_exchange()
application = initialize_telegram()

# 2. Global Ayarlar
CONFIG = {
    'SYMBOLS': [],
    'running': True,
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

# 3. LSTM Modeli
def create_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(CONFIG['LSTM_LOOKBACK'], 3)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

CONFIG['model'] = create_lstm_model()

# 4. Veri Yönetimi
async def fetch_ohlcv(symbol: str, timeframe: str):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        logging.error(f"{symbol} {timeframe} veri çekme hatası: {e}")
        return None

def calculate_indicators(df: pd.DataFrame):
    try:
        # EMA
        df['EMA9'] = df['close'].ewm(span=CONFIG['EMA_PERIODS'][0]).mean()
        df['EMA21'] = df['close'].ewm(span=CONFIG['EMA_PERIODS'][1]).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(CONFIG['RSI_PERIOD']).mean()
        avg_loss = loss.rolling(CONFIG['RSI_PERIOD']).mean()
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(CONFIG['ATR_PERIOD']).mean()
        
        return df.dropna()
    except Exception as e:
        logging.error(f"Gösterge hesaplama hatası: {e}")
        return df

# 5. Sinyal Üretimi
async def generate_signal(symbol: str):
    try:
        # Çoklu zaman dilimi verisi
        tf_data = {}
        for timeframe in CONFIG['TIMEFRAMES']:
            df = await fetch_ohlcv(symbol, timeframe)
            if df is not None:
                tf_data[timeframe] = calculate_indicators(df)

        # Temel sinyal mantığı
        df_5m = tf_data.get('5m')
        if df_5m is None or len(df_5m) < 10:
            return

        current_price = df_5m['close'].iloc[-1]
        last_ema9 = df_5m['EMA9'].iloc[-2:-1].values[0]
        last_ema21 = df_5m['EMA21'].iloc[-2:-1].values[0]
        
        long_signal = (last_ema9 < last_ema21) and (df_5m['EMA9'].iloc[-1] > df_5m['EMA21'].iloc[-1])
        short_signal = (last_ema9 > last_ema21) and (df_5m['EMA21'].iloc[-1] > df_5m['EMA9'].iloc[-1])

        if not (long_signal or short_signal):
            return

        # Risk yönetimi
        atr = df_5m['ATR'].iloc[-1]
        entry_price = current_price
        tp = entry_price + (2*atr) if long_signal else entry_price - (2*atr)
        sl = entry_price - atr if long_signal else entry_price + atr
        
        # Pozisyon boyutu
        balance = exchange.fetch_balance()['USDT']['free']
        risk_amount = balance * CONFIG['RISK_PER_TRADE']
        position_size = risk_amount / abs(entry_price - sl)

        # Sinyal kaydı
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
        await broadcast_signal(signal)
        
    except Exception as e:
        logging.error(f"{symbol} sinyal üretim hatası: {e}")

# 6. Telegram Entegrasyonu
async def broadcast_signal(signal: dict):
    message = (
        f"🚨 **Yeni Sinyal** 🚨\n"
        f"• Sembol: `{signal['symbol']}`\n"
        f"• Yön: {signal['direction']}\n"
        f"• Giriş: {signal['entry']:.4f}\n"
        f"• TP: {signal['tp']:.4f}\n"
        f"• SL: {signal['sl']:.4f}\n"
        f"• Boyut: {signal['size']:.2f} USDT"
    )
    
    for chat_id in CONFIG['chat_ids']:
        try:
            await application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logging.error(f"{chat_id} mesaj gönderme hatası: {e}")

# 7. Komutlar
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    CONFIG['chat_ids'].add(chat_id)
    await context.bot.send_message(
        chat_id=chat_id,
        text="✅ **Gate.io Trading Bot Aktif**\n"
             "Sinyaller burada görünecek.\n"
             "Komutlar:\n"
             "/stop - Botu durdur\n"
             "/signals - Aktif sinyalleri göster"
    )

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG['running'] = False
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🛑 Bot durduruluyor..."
    )

async def show_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CONFIG['signals']:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="ℹ️ Aktif sinyal bulunmuyor"
        )
        return
    
    response = ["📊 **Aktif Sinyaller**"]
    for signal in CONFIG['signals'].values():
        response.append(
            f"• {signal['symbol']} - {signal['direction']} "
            f"(Giriş: {signal['entry']:.4f})"
        )
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="\n".join(response)
    )

# 8. Ana Döngü
async def main():
    # Komutları kaydet
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("signals", show_signals))

    # Botu başlat
    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    # Sembolleri yükle
    markets = exchange.load_markets()
    CONFIG['SYMBOLS'] = [
        symbol for symbol, market in markets.items() 
        if market['type'] == 'swap' and market['active']
    ]
    logging.info(f"{len(CONFIG['SYMBOLS'])} adet sembol yüklendi")  # Düzeltildi

    # Sinyal üretim döngüsü
    while CONFIG['running']:
        try:
            for symbol in CONFIG['SYMBOLS'][:20]:  # Test için ilk 20 sembol
                await generate_signal(symbol)
            await asyncio.sleep(60)  # 1 dakikada bir kontrol
        except Exception as e:
            logging.error(f"Ana döngü hatası: {e}")
            await asyncio.sleep(10)

    # Temiz kapatma
    await application.updater.stop()
    await application.stop()
    await application.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
