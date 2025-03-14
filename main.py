import sys
import io
import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
import nest_asyncio
import aiosqlite
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from tensorflow.keras.models import load_model
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import os
import talib
import time

# 1. Sistem ve Unicode Ayarları
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
logger = logging.getLogger(__name__)
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
    
    @classmethod
    async def close(cls):
        if cls._conn:
            await cls._conn.close()

# 4. Exchange Konfigürasyonu
def get_exchange():
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True,
        },
        'timeout': 30000,
        'headers': {
            'User-Agent': 'Mozilla/5.0',
            'X-MBX-APIKEY': os.getenv('BINANCE_API_KEY', '')
        }
    })

# 5. Veri Yönetimi
class DataFetcher:
    def __init__(self):
        self.exchange = get_exchange()
    
    async def fetch_ohlcv_with_retry(self, symbol, timeframe, retries=3):
        for i in range(retries):
            try:
                # Zaman senkronizasyonu
                server_time = await self.exchange.fetch_time()
                local_time = self.exchange.milliseconds()
                time_diff = server_time - local_time
                
                return await self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=100,
                    params={'timestamp': local_time + time_diff}
                )
            except Exception as e:
                if i == retries - 1:
                    raise
                await asyncio.sleep(2 ** i)
        return None

    async def process_and_save_data(self, symbol, timeframe):
        try:
            ohlcv = await self.fetch_ohlcv_with_retry(symbol, timeframe)
            if not ohlcv:
                return False

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
            logger.error(f"Veri işleme hatası ({symbol} {timeframe}): {str(e)}")
            return False

# 6. Sinyal Üretimi
class SignalGenerator:
    def __init__(self):
        self.exchange = get_exchange()
    
    async def check_conditions(self, symbol):
        try:
            conn = await Database.get_connection()
            
            # Veri kontrolü
            count = await conn.execute_fetchall(
                'SELECT COUNT(*) FROM market_data WHERE symbol=?', 
                (symbol,)
            )
            if count[0][0] < 100:
                logger.warning(f"{symbol}: Yetersiz veri ({count[0][0]}/100)")
                return None

            # Son 2 mum verisi
            data = await conn.execute_fetchall('''
                SELECT * FROM market_data 
                WHERE symbol=? AND timeframe='5m' 
                ORDER BY timestamp DESC LIMIT 2''', 
                (symbol,)
            )
            
            if len(data) < 2:
                return None

            prev, current = data[1], data[0]

            # EMA ve RSI koşulları
            ema_cross = (prev[8] < prev[9] and current[8] > current[9]) or \
                        (prev[8] > prev[9] and current[8] < current[9])
            rsi_ok = (current[10] < 65) if current[8] > current[9] else (current[10] > 35)
            
            if not (ema_cross and rsi_ok):
                return None

            # Model kontrolü
            model_path = f"models/{symbol.replace('/', '_')}_lstm.keras"
            if not os.path.isfile(model_path):
                logger.error(f"Model bulunamadı: {model_path}")
                return None

            # Veri hazırlama
            X = np.array([row[3:9] for row in await conn.execute_fetchall('''
                SELECT open, high, low, close, volume, ema9 
                FROM market_data 
                WHERE symbol=? 
                ORDER BY timestamp DESC LIMIT 60''', 
                (symbol,))]
            ).reshape(1, 60, 6)
            
            # Tahmin
            model = load_model(model_path)
            prediction = model.predict(X, verbose=0)[0][0]
            
            if prediction < 0.65:
                return None

            # Risk yönetimi
            balance = (await self.exchange.fetch_balance(params={'type':'future'}))['USDT']['free']
            atr = current[11]
            size = round((balance * 0.02) / (atr * 1.5), 4)
            
            return {
                'symbol': symbol,
                'direction': 'LONG' if current[8] > current[9] else 'SHORT',
                'entry': round(current[3], 4),
                'tp': round(current[3] + (2*atr), 4) if current[8] > current[9] else round(current[3] - (2*atr), 4),
                'sl': round(current[3] - atr, 4) if current[8] > current[9] else round(current[3] + atr, 4),
                'size': size,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Sinyal hatası ({symbol}): {str(e)}")
            return None

# 7. Telegram Bot Yönetimi
class TradingBot:
    def __init__(self):
        self.app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
        self.scheduler = AsyncIOScheduler()
        self.symbols = []
        self.active = False
        self.data_fetcher = DataFetcher()
        self.signal_generator = SignalGenerator()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.active = True
        await update.message.reply_text("🚀 Trading Bot Aktif!")
        logger.info("Bot kullanıcı tarafından başlatıldı")

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.active = False
        await update.message.reply_text("🔴 Trading Bot Durduruldu!")
        logger.info("Bot kullanıcı tarafından durduruldu")

    async def broadcast_signal(self, signal):
        message = (
            f"⚡️ **{signal['symbol']} {signal['direction']} Sinyali**\n"
            f"▫️ Giriş: `{signal['entry']}`\n"
            f"▫️ TP: `{signal['tp']}`\n"
            f"▫️ SL: `{signal['sl']}`\n"
            f"▫️ Miktar: `{signal['size']}`"
        )
        async with self.app:
            await self.app.bot.send_message(
                chat_id=update.message.chat_id,
                text=message,
                parse_mode='Markdown'
            )

    async def market_scan(self):
        if not self.active:
            return
            
        logger.info("Piyasa taraması başlatılıyor...")
        for symbol in self.symbols:
            try:
                # Veri güncelleme
                logger.info(f"{symbol} verileri güncelleniyor...")
                await self.data_fetcher.process_and_save_data(symbol, '5m')
                await self.data_fetcher.process_and_save_data(symbol, '15m')
                
                # Sinyal kontrolü
                logger.info(f"{symbol} sinyal kontrolü...")
                if signal := await self.signal_generator.check_conditions(symbol):
                    await self.broadcast_signal(signal)
                    logger.info(f"Yeni sinyal: {signal}")
                    
            except Exception as e:
                logger.error(f"Tarama hatası ({symbol}): {str(e)}")

    async def run(self):
        try:
            # Sembolleri yükle
            self.symbols = await self.load_symbols()
            
            # Handler'ları ayarla
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CommandHandler("stop", self.stop))
            
            # Zamanlayıcıyı ayarla
            self.scheduler.add_job(self.market_scan, 'interval', minutes=5)
            self.scheduler.start()
            
            logger.info("Bot başlatılıyor...")
            await self.app.run_polling()
        finally:
            await Database.close()
            await self.exchange.close()

    async def load_symbols(self):
        try:
            markets = await get_exchange().load_markets()
            return [
                s for s in markets 
                if s.endswith('/USDT') 
                and markets[s].get('future')
                and markets[s]['active']
            ][:5]
        except:
            return ['BTC/USDT', 'ETH/USDT']

# 8. Ana Yürütme Bloğu
if __name__ == '__main__':
    # Sistem saati kontrolü
    if datetime.now().year > 2024:
        logger.critical("Sistem saati hatalı! Lütfen kontrol edin.")
        sys.exit(1)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = TradingBot()
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("Güvenli çıkış yapılıyor...")
    except Exception as e:
        logger.critical(f"Kritik hata: {str(e)}")
    finally:
        if loop.is_running():
            loop.close()
