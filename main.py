import sys
import io
import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
import nest_asyncio
import aiosqlite
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from tensorflow.keras.models import load_model
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import os
import talib

# 1. Unicode ve Sistem Ayarları
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
nest_asyncio.apply()

# 2. Logging Konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
load_dotenv('config.env')

# 3. Veritabanı Yönetimi
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
        await cls._conn.execute('''CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT, timeframe TEXT, timestamp DATETIME,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            ema9 REAL, ema21 REAL, rsi REAL, atr REAL,
            PRIMARY KEY (symbol, timeframe, timestamp))''')
        await cls._conn.commit()

# 4. Binance Futures Konfigürasyonu
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
    'timeout': 30000,
    'headers': {'User-Agent': 'Mozilla/5.0'}
})

# 5. Sembol Yönetimi
async def get_active_symbols():
    try:
        markets = await exchange.load_markets()
        return [
            s for s in markets 
            if s.endswith('/USDT') 
            and markets[s].get('future')
            and markets[s]['active']
        ][:5]  # Test için ilk 5 sembol
    except Exception as e:
        logging.error(f"Sembol hatası: {str(e)}")
        return ['BTC/USDT', 'ETH/USDT']

# 6. Veri Yönetimi
async def fetch_and_save_data(symbol, timeframe):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=100)
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
            INSERT OR REPLACE INTO market_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            [(symbol, timeframe, row.timestamp, row.open, row.high, row.low,
              row.close, row.volume, row.ema9, row.ema21, row.rsi, row.atr)
             for row in df.dropna().itertuples()])
        await conn.commit()
        return True
    except Exception as e:
        logging.error(f"Veri hatası ({symbol} {timeframe}): {str(e)}")
        return False

# 7. Sinyal Üretimi
async def generate_signal(symbol):
    try:
        conn = await Database.get_connection()
        
        # Son 2 mum verisi
        data = await conn.execute_fetchall('''
            SELECT * FROM market_data 
            WHERE symbol=? AND timeframe='5m' 
            ORDER BY timestamp DESC LIMIT 2''', (symbol,))
        
        if len(data) < 2:
            return None

        prev = data[1]
        current = data[0]

        # EMA Kesişim ve RSI Kontrolü
        ema_cross = (prev[8] < prev[9] and current[8] > current[9]) or \
                    (prev[8] > prev[9] and current[8] < current[9])
        rsi_ok = (current[10] < 65) if current[8] > current[9] else (current[10] > 35)
        
        if ema_cross and rsi_ok:
            # Model Yükleme
            model_path = f"models/{symbol.replace('/', '_')}_lstm.keras"
            if not os.path.exists(model_path):
                logging.error(f"Model bulunamadı: {model_path}")
                return None
            
            # Veri Hazırlama
            X = np.array([row[3:9] for row in await conn.execute_fetchall('''
                SELECT open, high, low, close, volume, ema9 
                FROM market_data 
                WHERE symbol=? 
                ORDER BY timestamp DESC LIMIT 60''', (symbol,))]).reshape(1, 60, 6)
            
            # Tahmin
            prediction = load_model(model_path).predict(X, verbose=0)[0][0]
            if prediction > 0.65:
                # Pozisyon Hesaplama
                balance = (await exchange.fetch_balance())['USDT']['free']
                atr = current[11]
                size = round((balance * 0.02) / (atr * 1.5), 4)
                
                return {
                    'symbol': symbol,
                    'direction': 'LONG' if current[8] > current[9] else 'SHORT',
                    'entry': round(current[3], 4),
                    'tp': round(current[3] + (2*atr), 4) if current[8] > current[9] else round(current[3] - (2*atr), 4),
                    'sl': round(current[3] - atr, 4) if current[8] > current[9] else round(current[3] + atr, 4),
                    'size': size
                }
        return None
    except Exception as e:
        logging.error(f"Sinyal hatası ({symbol}): {str(e)}")
        return None

# 8. Telegram Bot Yönetimi
class TradingBot:
    def __init__(self):
        self.app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
        self.scheduler = AsyncIOScheduler()
        self.symbols = []
        self.active = False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.active = True
        await update.message.reply_text("🚀 Trading Bot Aktif!")
        logging.info("Kullanıcı botu başlattı")

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.active = False
        await update.message.reply_text("🔴 Trading Bot Durduruldu!")
        logging.info("Kullanıcı botu durdurdu")

    async def broadcast_signal(self, signal):
        message = (
            f"📈 {signal['symbol']} {signal['direction']}\n"
            f"Giriş: {signal['entry']}\n"
            f"TP: {signal['tp']}\n"
            f"SL: {signal['sl']}\n"
            f"Miktar: {signal['size']}"
        )
        async with self.app:
            await self.app.bot.send_message(chat_id=1413321448, text=message)

    async def market_scan(self):
        if not self.active:
            return
            
        for symbol in self.symbols:
            try:
                # Veri Güncelleme
                await fetch_and_save_data(symbol, '5m')
                await fetch_and_save_data(symbol, '15m')
                
                # Sinyal Üretme
                if signal := await generate_signal(symbol):
                    await self.broadcast_signal(signal)
                    logging.info(f"Yeni sinyal: {signal}")
                    
            except Exception as e:
                logging.error(f"Tarama hatası ({symbol}): {str(e)}")

    async def run(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("stop", self.stop))
        
        self.symbols = await get_active_symbols()
        self.scheduler.add_job(self.market_scan, 'interval', minutes=5)
        self.scheduler.start()
        
        logging.info("Bot başlatılıyor...")
        await self.app.run_polling()

if __name__ == '__main__':
    try:
        bot = TradingBot()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logging.info("Bot güvenli şekilde kapatılıyor...")
    finally:
        loop.close()
