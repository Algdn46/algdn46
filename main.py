import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
import aiosqlite
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import os
import talib

# 1. Logging ve Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trader.log'), logging.StreamHandler()]
)
load_dotenv('config.env')

# 2. Async Veritabanı Yönetimi
class Database:
    _conn = None
    
    @classmethod
    async def get_connection(cls):
        if not cls._conn:
            cls._conn = await aiosqlite.connect('trading.db')
            await cls.initialize_db()
        return cls._conn
    
    @classmethod
    async def initialize_db(cls):
        async with cls._conn.execute('''CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT, timeframe TEXT, timestamp DATETIME PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            ema9 REAL, ema21 REAL, rsi REAL, atr REAL)'''):
            await cls._conn.commit()

# 3. Exchange Konfigürasyonu
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'timeout': 20000
})

# 4. Sembol Yönetimi
async def get_active_futures():
    try:
        markets = await exchange.load_markets()
        return [
            s for s in markets 
            if s.endswith('/USDT') 
            and markets[s].get('future')
            and markets[s]['active']
        ][:5]  # Test için 5 sembolle sınırla
    except Exception as e:
        logging.error(f"Sembol hatası: {e}")
        return ['BTC/USDT', 'ETH/USDT']

# 5. Veri Yönetimi
async def fetch_and_store(symbol, timeframe):
    for _ in range(3):  # 3 deneme
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Teknik Göstergeler
            df['ema9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema21'] = talib.EMA(df['close'], timeperiod=21)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Veritabanına Kaydet
            conn = await Database.get_connection()
            await conn.executemany('''
                INSERT OR REPLACE INTO market_data VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(symbol, timeframe, row.timestamp, row.open, row.high, row.low, 
                  row.close, row.volume, row.ema9, row.ema21, row.rsi, row.atr)
                  for row in df.dropna().itertuples()])
            await conn.commit()
            return True
        except ccxt.NetworkError:
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Veri hatası ({symbol} {timeframe}): {e}")
            return False
    return False

# 6. Sinyal Üretimi
async def generate_signal(symbol):
    try:
        conn = await Database.get_connection()
        # Son 2 mum verisi
        data = await conn.execute_fetchall('''
            SELECT * FROM market_data 
            WHERE symbol = ? AND timeframe = '5m'
            ORDER BY timestamp DESC LIMIT 2
        ''', (symbol,))
        
        if len(data) < 2:
            return None

        current = data[0]
        prev = data[1]

        # EMA Kesişim Kontrolü
        ema_cross = (prev[8] < prev[9] and current[8] > current[9]) or \
                    (prev[8] > prev[9] and current[8] < current[9])
        
        # RSI Filtresi
        rsi_ok = (current[10] < 65) if current[8] > current[9] else (current[10] > 35)
        
        if ema_cross and rsi_ok:
            # Model Validasyonu
            model = load_model(f'models/{symbol.replace("/", "_")}_lstm.keras')
            X = np.array([row[3:9] for row in await conn.execute_fetchall('''
                SELECT open, high, low, close, volume, ema9, ema21, rsi, atr 
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC LIMIT 60
            ''', (symbol,))]).reshape(1, 60, -1)
            
            if model.predict(X, verbose=0)[0][0] > 0.65:
                # Pozisyon Hesapla
                balance = (await exchange.fetch_balance())['USDT']['free']
                atr = current[11]
                size = round((balance * 0.02) / (atr * 1.5), 4)
                
                return {
                    'symbol': symbol,
                    'direction': 'LONG' if current[8] > current[9] else 'SHORT',
                    'entry': current[3],
                    'tp': current[3] + (2*atr) if current[8] > current[9] else current[3] - (2*atr),
                    'sl': current[3] - atr if current[8] > current[9] else current[3] + atr,
                    'size': size
                }
        return None
    except Exception as e:
        logging.error(f"Sinyal hatası ({symbol}): {e}")
        return None

# 7. Telegram Bot
class TradingBot:
    def __init__(self):
        self.app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
        self.scheduler = AsyncIOScheduler()
        self.symbols = []
        self.running = False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.running = True
        await update.message.reply_text("🤖 Bot aktif! Piyasa taranıyor...")
        
    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.running = False
        await update.message.reply_text("🛑 Bot durduruldu!")

    async def broadcast(self, signal):
        msg = (
            f"🚨 {signal['symbol']} {signal['direction']}\n"
            f"Giriş: {signal['entry']:.4f}\n"
            f"TP: {signal['tp']:.4f}\n"
            f"SL: {signal['sl']:.4f}\n"
            f"Miktar: {signal['size']}"
        )
        async with self.app:
            for chat_id in set(await Database.get_connection().execute_fetchall("SELECT chat_id FROM users")):
                await self.app.bot.send_message(chat_id[0], msg)

    async def market_scan(self):
        if not self.running:
            return
            
        for symbol in self.symbols:
            try:
                # Verileri Güncelle
                for timeframe in ['5m', '15m', '1h']:
                    await fetch_and_store(symbol, timeframe)
                
                # Sinyal Kontrolü
                if signal := await generate_signal(symbol):
                    await self.broadcast(signal)
                    logging.info(f"Yeni sinyal: {signal}")
                    
            except Exception as e:
                logging.error(f"Tarama hatası ({symbol}): {e}")

    async def run(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("stop", self.stop))
        
        self.symbols = await get_active_futures()
        self.scheduler.add_job(self.market_scan, 'interval', minutes=5)
        self.scheduler.start()
        
        logging.info("Bot başlatılıyor...")
        await self.app.run_polling()

if __name__ == '__main__':
    bot = TradingBot()
    asyncio.run(bot.run())
