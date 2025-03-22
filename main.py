import os
import pickle
import logging
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import ccxt.async_support as ccxt
import pandas_ta as ta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue
from flask import Flask
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import subprocess
import pytz
from datetime import datetime
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv('config.env')

# Ortam değişkenlerinden Telegram token ve chat ID'yi al
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# Hata kontrolü
if TELEGRAM_TOKEN is None:
    raise ValueError("Hata: TELEGRAM_TOKEN 'config.env' dosyasında tanımlı değil!")
if CHAT_ID is None:
    raise ValueError("Hata: CHAT_ID 'config.env' dosyasında tanımlı değil!")
CHAT_ID = int(CHAT_ID)  # CHAT_ID bir tamsayı olmalı

# Ortak değişkenler
LOOKBACK = 60
INTERVAL = '5m'
TR_TIMEZONE = pytz.timezone('Europe/Istanbul')
last_signals = {}
last_signal_times = {}

# Logging yapılandırması
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,  # INFO yerine DEBUG
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask uygulaması (self-ping için)
app = Flask(__name__)

@app.route('/')
async def keep_alive():
    return "Bot çalışıyor!"

# Exchange bağlantısı
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Özel JobQueue sınıfı
class CustomJobQueue(JobQueue):
    def __init__(self):
        # JobQueue'un temel özelliklerini manuel olarak başlatıyoruz
        self._application = None
        self._is_running = False
        # Bizim scheduler'ımızı oluşturuyoruz
        self._scheduler = AsyncIOScheduler(timezone=TR_TIMEZONE)
        self.scheduler = self._scheduler

    def set_application(self, application):
        self._application = application

    def start(self):
        if not self._is_running:
            self._is_running = True
            self.scheduler.start()
            logger.info("CustomJobQueue scheduler started")

    def stop(self):
        if self._is_running:
            self._is_running = False
            self.scheduler.shutdown()
            logger.info("CustomJobQueue scheduler stopped")

# Model oluşturma
def create_enhanced_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.0001, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model

# Veri işleme
def preprocess_data(df):
    logger.info("preprocess_data fonksiyonu çağrıldı.")
    
    logger.info(f"Veri çerçevesi sütunları: {df.columns.tolist()}")
    logger.info("Veri çerçevesi ilk 5 satır:\n" + df.head().to_string())
    
    df = df.drop(columns=['timestamp'])
    
    for column in ['open', 'high', 'low', 'close', 'volume']:
        logger.info(f"{column} sütunu tipi: {df[column].dtype}")
        df[column] = df[column].astype(float)
    
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['bollinger'] = ta.volatility.bollinger_hband_indicator(df['close'], window=20, window_dev=2)
    
    for column in ['ma7', 'ma21', 'rsi', 'macd', 'bollinger']:
        logger.info(f"{column} sütunu tipi: {df[column].dtype}")
        df[column] = df[column].astype(float)
    
    df = df.dropna()
    
    scaler = pickle.load(open('model_repo/scalers/global_scaler.pkl', 'rb')) if os.path.exists('model_repo/scalers/global_scaler.pkl') else None
    if not scaler:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df)
        with open('model_repo/scalers/global_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    scaled_data = scaler.transform(df)
    logger.info("Veri başarıyla ölçeklendirildi.")
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i])
        y.append(scaled_data[i, 3])
    
    return np.array(X), np.array(y)

# Model yönetimi
class ModelManager:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None
        self.scaler = None

    async def load_model(self):
        model_path = os.path.join("model_repo", "models", f"{self.symbol.replace('/', '_')}.keras")
        scaler_path = os.path.join("model_repo", "scalers", f"{self.symbol.replace('/', '_')}.pkl")
        
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"{self.symbol} | Model yüklendi.")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"{self.symbol} | Scaler yüklendi.")

    async def train_model(self, ohlcv):
        try:
            logger.info(f"{self.symbol} | OHLCV verisi: {len(ohlcv)} satır")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
            
            logger.info(f"{self.symbol} | preprocess_data fonksiyonu çağrılıyor...")
            X, y = preprocess_data(df)
            
            logger.info(f"{self.symbol} | X veri tipi: {X.dtype}, şekli: {X.shape}")
            logger.info(f"{self.symbol} | y veri tipi: {y.dtype}, şekli: {y.shape}")
            
            checkpoint_path = os.path.join("model_repo", "checkpoints", f"{self.symbol.replace('/', '_')}_checkpoint.keras")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
            
            self.model = create_enhanced_model((LOOKBACK, X.shape[2]))
            history = self.model.fit(X, y, epochs=75, batch_size=64, validation_split=0.2,
                                    callbacks=[checkpoint, EarlyStopping(patience=5)], verbose=0)
            
            self.model.save(os.path.join("model_repo", "models", f"{self.symbol.replace('/', '_')}.keras"))
            with open(os.path.join("model_repo", "scalers", f"{self.symbol.replace('/', '_')}.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(os.path.join("model_repo", "training_history", f"{self.symbol.replace('/', '_')}_history.pkl"), 'wb') as f:
                pickle.dump(history.history, f)
            
            await git_push(f"Model ve scaler güncellendi: {self.symbol}")
        except Exception as e:
            logger.error(f"{self.symbol} | Model eğitimi hatası: {str(e)}", exc_info=True)
            raise

# Git push işlemi
async def git_push(message):
    try:
        subprocess.run(["git", "add", "model_repo"], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logger.info("Git push işlemi tamamlandı.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git push hatası: {str(e)}")

# Fiyat yuvarlama
def round_price(price, symbol):
    info = exchange.markets[symbol]
    tick_size = info['precision']['price']
    return round(price / tick_size) * tick_size

# Sinyal üretimi (Anlık veri entegrasyonu eklendi)
async def generate_signal(symbol, manager, news_sentiment):
    try:
        if not manager.model or not manager.scaler:
            logger.error(f"{symbol} | Model veya scaler yüklenemedi.")
            return None, None, None, None
        
        # Anlık fiyatı al
        logger.info(f"{symbol} | Anlık fiyat çekiliyor...")
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker['last']  # Anlık fiyat
        logger.info(f"{symbol} | Anlık fiyat: {current_price}")

        # OHLCV verisi çek (geçmiş veriler için)
        logger.info(f"{symbol} | OHLCV verisi çekiliyor...")
        ohlcv = await exchange.fetch_ohlcv(symbol, INTERVAL, limit=LOOKBACK+14)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TR_TIMEZONE)
        
        logger.info(f"{symbol} | Veriyi preprocess_data ile işleme...")
        data, _ = preprocess_data(df)
        
        logger.info(f"{symbol} | data şekli: {data.shape}")
        if data.shape[0] < LOOKBACK:
            logger.error(f"{symbol} | Yeterli veri yok, data uzunluğu: {data.shape[0]}, gereken: {LOOKBACK}")
            return None, None, None, None
        
        X = data[-LOOKBACK:]
        X = np.array([X])
        logger.info(f"{symbol} | X şekli: {X.shape}")
        
        expected_shape = (1, LOOKBACK, 10)
        if X.shape != expected_shape:
            logger.error(f"{symbol} | X şekli uyumsuz, beklenen: {expected_shape}, bulunan: {X.shape}")
            return None, None, None, None
        
        logger.info(f"{symbol} | Tahmin yapılıyor...")
        prediction = manager.model.predict(X, verbose=0)[0][0]
        
        # ATR ve diğer göstergeler için OHLCV verisini kullan
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).iloc[-1]
        
        # Trend belirleme (hafif gevşetilmiş)
        last_5_closes = df['close'].tail(5).values
        avg_closes = np.mean(last_5_closes)
        if last_5_closes[-1] > avg_closes * 1.001:
            trend_direction = 'UP'
        elif last_5_closes[-1] < avg_closes * 0.999:
            trend_direction = 'DOWN'
        else:
            trend_direction = 'NEUTRAL'
        logger.info(f"{symbol} | Trend yönü: {trend_direction}")
        
        # ATR ile volatilite kontrolü
        if atr < current_price * 0.002:
            logger.info(f"{symbol} | ATR çok düşük ({atr}), trend NEUTRAL olarak işaretlendi.")
            trend_direction = 'NEUTRAL'
        
        # Hacim onayı (minimum gevşetme)
        avg_volume = df['volume'].rolling(window=14).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_confirmed = current_volume > avg_volume * 1.8  # 2 yerine 1.8
        if not volume_confirmed:
            last_3_volumes = df['volume'].tail(3).values
            if len(last_3_volumes) == 3 and all(last_3_volumes[i] < last_3_volumes[i+1] for i in range(len(last_3_volumes)-1)):
                volume_confirmed = True
        if not volume_confirmed:
            logger.info(f"{symbol} | Hacim onayı başarısız.")
            return None, None, None, None
        
        # RSI kontrolü
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
        if not (30 < rsi < 70):
            logger.info(f"{symbol} | RSI uygun değil: {rsi}")
            return None, None, None, None
        
        # MACD ile trend onayı
        macd = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9).iloc[-1]
        if trend_direction == 'UP' and macd <= 0:
            logger.info(f"{symbol} | MACD trendle uyumsuz ({macd}), sinyal engellendi.")
            return None, None, None, None
        elif trend_direction == 'DOWN' and macd >= 0:
            logger.info(f"{symbol} | MACD trendle uyumsuz ({macd}), sinyal engellendi.")
            return None, None, None, None
        
        # Haber sentiment kontrolü
        if news_sentiment < -0.3 and trend_direction == 'UP':
            logger.info(f"{symbol} | Negatif haber sentiment ({news_sentiment}) nedeniyle LONG sinyali engellendi.")
            return None, None, None, None
        elif news_sentiment > 0.3 and trend_direction == 'DOWN':
            logger.info(f"{symbol} | Pozitif haber sentiment ({news_sentiment}) nedeniyle SHORT sinyali engellendi.")
            return None, None, None, None
        
        # Önceki sinyal kontrolü
        if symbol in last_signals:
            last_signal = last_signals[symbol]
            last_direction, last_entry, last_sl, last_tp = last_signal
            last_tp_max = max(last_tp) if last_tp else last_entry
            if prediction > current_price * 1.002 and last_direction == 'LONG':
                if current_price < last_tp_max:
                    logger.info(f"{symbol} | Önceki LONG sinyali nedeniyle atlanıyor.")
                    return None, None, None, None
            elif prediction < current_price * 0.998 and last_direction == 'SHORT':
                if current_price > last_tp_max:
                    logger.info(f"{symbol} | Önceki SHORT sinyali nedeniyle atlanıyor.")
                    return None, None, None, None
        
        # Sinyal üretimi (Anlık fiyatla karşılaştırma)
        price_threshold = 0.005
        if news_sentiment > 0:
            price_threshold -= news_sentiment * 0.5
        elif news_sentiment < 0:
            price_threshold += news_sentiment * 0.5
        
        if prediction > current_price * (1 + price_threshold) and trend_direction == 'UP':
            sl = current_price - (atr * 1.5)
            tp = [current_price + (atr * i) for i in [2, 3, 5]]
            tp_levels = [round_price(t, symbol) for t in tp if prediction >= t]
            if not tp_levels:
                logger.info(f"{symbol} | TP seviyeleri uygun değil.")
                return None, None, None, None
            logger.info(f"{symbol} | LONG sinyali üretildi.")
            return 'LONG', current_price, round_price(sl, symbol), tuple(tp_levels)
        
        elif prediction < current_price * (1 - price_threshold) and trend_direction == 'DOWN':
            sl = current_price + (atr * 1.5)
            tp = [current_price - (atr * i) for i in [2, 3, 5]]
            tp_levels = [round_price(t, symbol) for t in tp if prediction <= t]
            if not tp_levels:
                logger.info(f"{symbol} | TP seviyeleri uygun değil.")
                return None, None, None, None
            logger.info(f"{symbol} | SHORT sinyali üretildi.")
            return 'SHORT', current_price, round_price(sl, symbol), tuple(tp_levels)
        
        logger.info(f"{symbol} | Sinyal üretilemedi.")
        return None, None, None, None
    except Exception as e:
        logger.error(f"{symbol} | Hata: {str(e)}", exc_info=True)
        return None, None, None, None

# Haber sentiment analizi (basitleştirilmiş)
async def fetch_news():
    return [{'title': 'Sample news', 'description': 'Positive news'}]

async def analyze_news_sentiment(news):
    return 0.1  # Basitleştirilmiş

# Top gainer belirleme
async def get_top_gainer(symbols):
    top_gainer = None
    max_change = -float('inf')
    
    for symbol in symbols:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, '1h', limit=24)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            price_change = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0] * 100
            if price_change > max_change:
                max_change = price_change
                top_gainer = symbol
        except Exception as e:
            logger.error(f"{symbol} | Top gainer hatası: {str(e)}")
            continue
    
    return top_gainer

# Sembol tarama
async def scan_symbols(context: ContextTypes.DEFAULT_TYPE):
    logger.info("scan_symbols fonksiyonu çağrılıyor...")
    logger.info("Sinyaller taranıyor...")
    
    for attempt in range(1, 4):
        logger.info(f"load_markets denemesi: {attempt}")
        try:
            markets = await exchange.load_markets()
            break
        except Exception as e:
            logger.error(f"load_markets hatası: {str(e)}")
            if attempt == 3:
                logger.error("load_markets başarısız, işlem durduruluyor.")
                return
            await asyncio.sleep(5)
    
    logger.info("Markets yüklendi.")
    symbols = [s for s in markets if markets[s]['type'] == 'spot' and markets[s]['active'] and 'USDT' in s]
    
    logger.info("Haberler çekiliyor...")
    news = await fetch_news()
    logger.info("Haber sentiment analizi yapılıyor...")
    news_sentiment = await analyze_news_sentiment(news)
    
    logger.info("Top gainer belirleniyor...")
    top_gainer = await get_top_gainer(symbols)
    logger.info(f"Top gainer: {top_gainer}")
    
    logger.info(f"Toplam sembol sayısı: {len(symbols)}")
    
    found_signal = False
    ALLOWED_GROUP_CHAT_ID = CHAT_ID  # Daha önce kullanılan değişkeni güncelliyoruz
    
    logger.info(f"Top gainer ({top_gainer}) için tarama başlıyor...")
    try:
        logger.info(f"{top_gainer} için yeni ModelManager oluşturuluyor...")
        manager = ModelManager(top_gainer)
        await manager.load_model()
        
        if not manager.model:
            logger.info(f"{top_gainer} için OHLCV verisi çekiliyor...")
            ohlcv = await exchange.fetch_ohlcv(top_gainer, INTERVAL, limit=300)  # limit=200 yerine 300
            logger.info(f"{top_gainer} için model eğitiliyor...")
            await manager.train_model(ohlcv)
        
        direction, entry, sl, tp = await generate_signal(top_gainer, manager, news_sentiment)
        if direction:
            current_signal = (direction, entry, sl, tp)
            message = f"🚨 Sinyal: {top_gainer}\nYön: {direction}\nGiriş: {entry}\nSL: {sl}\nTP: {tp}"
            await context.bot.send_message(
                chat_id=ALLOWED_GROUP_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            logger.info(f"Sinyal gönderildi (chat_id: {ALLOWED_GROUP_CHAT_ID}): {message}")
            last_signals[top_gainer] = current_signal
            last_signal_times[top_gainer] = datetime.now(TR_TIMEZONE)
            found_signal = True
            await asyncio.sleep(1)  # time.sleep yerine asyncio.sleep kullanıyoruz
    except Exception as e:
        logger.error(f"{top_gainer} | Hata: {str(e)}", exc_info=True)
    
    logger.info("Diğer semboller için tarama başlıyor...")
    for symbol in symbols:
        if symbol == top_gainer:
            continue
        try:
            logger.info(f"{symbol} için tarama yapılıyor...")
            logger.info(f"{symbol} için yeni ModelManager oluşturuluyor...")
            manager = ModelManager(symbol)
            await manager.load_model()
            
            if not manager.model:
                logger.info(f"{symbol} için OHLCV verisi çekiliyor...")
                ohlcv = await exchange.fetch_ohlcv(symbol, INTERVAL, limit=300)  # limit=200 yerine 300
                logger.info(f"{symbol} için model eğitiliyor...")
                await manager.train_model(ohlcv)
            
            direction, entry, sl, tp = await generate_signal(symbol, manager, news_sentiment)
            if direction:
                current_signal = (direction, entry, sl, tp)
                message = f"🚨 Sinyal: {symbol}\nYön: {direction}\nGiriş: {entry}\nSL: {sl}\nTP: {tp}"
                await context.bot.send_message(
                    chat_id=ALLOWED_GROUP_CHAT_ID,
                    text=message,
                    parse_mode='HTML'
                )
                logger.info(f"Sinyal gönderildi (chat_id: {ALLOWED_GROUP_CHAT_ID}): {message}")
                last_signals[symbol] = current_signal
                last_signal_times[symbol] = datetime.now(TR_TIMEZONE)
                found_signal = True
                await asyncio.sleep(1)  # time.sleep yerine asyncio.sleep kullanıyoruz
        except Exception as e:
            logger.error(f"{symbol} | Hata: {str(e)}", exc_info=True)
            continue

# Telegram komutları
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    logger.info(f"Start komutu alındı, chat_id: {chat_id}, chat_type: {chat_type}")
    
    logger.info("Kemerini tak mesajı gönderiliyor...")
    await context.bot.send_message(chat_id=chat_id, text="Kemerini tak dostum, sinyaller geliyor...")
    
    await scan_symbols(context)
    
    if not context.job_queue.get_jobs_by_name("continuous_scan"):
        context.job_queue.run_repeating(scan_symbols, interval=300, first=10, name="continuous_scan")
        logger.info("Sürekli tarama başlatıldı.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    logger.info(f"Stop komutu alındı, chat_id: {chat_id}")
    
    for job in context.job_queue.jobs():
        if job.name == "continuous_scan":
            job.schedule_removal()
            logger.info("continuous_scan işi kaldırıldı.")
    
    await context.bot.send_message(chat_id=chat_id, text="Bot durduruldu, iyi şanslar!")
    await context.application.job_queue.stop()
    logger.info("Job queue durduruldu.")

# Self-ping fonksiyonu
async def self_ping(context: ContextTypes.DEFAULT_TYPE):
    async with aiohttp.ClientSession() as session:
        async with session.get("http://127.0.0.1:10000") as response:
            if response.status == 200:
                logger.info("Self-ping gönderildi, bot uyanık tutuluyor...")
            else:
                logger.error("Self-ping başarısız!")

# Ana fonksiyon
async def main():
    logger.info("Bot başlatılıyor...")
    
    # CustomJobQueue oluştur
    job_queue = CustomJobQueue()
    
    # Application nesnesini oluştururken job_queue'u None olarak başlat
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # JobQueue'u manuel olarak sıfırla ve bizim CustomJobQueue'u bağla
    application._job_queue = None  # Varsayılan job_queue'u sıfırlıyoruz
    application.job_queue = job_queue
    job_queue.set_application(application)
    
    # JobQueue'u başlat
    job_queue.start()
    
    # Self-ping işini ekle
    application.job_queue.run_repeating(self_ping, interval=300, first=10)
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    
    await application.initialize()
    await application.start()
    logger.info("Application started")
    
    # Güncellemeleri almaya başla
    logger.info("Polling başlatılıyor...")
    await application.updater.start_polling(drop_pending_updates=True)
    logger.info("Polling başlatıldı, komutlar bekleniyor...")
    
    # Manuel olarak scan_symbols fonksiyonunu çalıştır
    logger.info("Manuel olarak scan_symbols çalıştırılıyor...")
    await scan_symbols(application)
    
    loop = asyncio.get_event_loop()
    loop.create_task(app.run(host='0.0.0.0', port=10000))

if __name__ == "__main__":
    asyncio.run(main())
