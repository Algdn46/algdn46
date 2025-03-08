import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
import nest_asyncio
import sqlite3
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import os
import joblib
import talib
import tensorflow as tf
from collections import deque

# Async sorunları için
nest_asyncio.apply()

# Logging Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ai_trading.log'), logging.StreamHandler()]
)

# Çevresel Değişkenler
load_dotenv('andirin.env')  # .env dosyası adı açıkça andirin.env olarak belirtildi

# Veritabanı Bağlantısı
conn = sqlite3.connect('/data/trading_data.db', check_same_thread=False)
c = conn.cursor()

# Veritabanı Tabloları
c.execute('''CREATE TABLE IF NOT EXISTS market_data
             (symbol TEXT, timeframe TEXT, timestamp DATETIME,
              open REAL, high REAL, low REAL, close REAL, volume REAL,
              ema9 REAL, ema21 REAL, rsi REAL, atr REAL,
              PRIMARY KEY (symbol, timeframe, timestamp))''')
c.execute('''CREATE TABLE IF NOT EXISTS model_performance
             (symbol TEXT, model_type TEXT, timestamp DATETIME,
              accuracy REAL, profit REAL)''')
conn.commit()

# Global Exchange Nesnesi
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# Tüm USDT Vadeli İşlem Çiftlerini Çekme
def load_usdt_futures_symbols():
    try:
        markets = exchange.load_markets()
        usdt_symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
        logging.info(f"{len(usdt_symbols)} adet USDT vadeli işlem çifti bulundu.")
        return usdt_symbols
    except Exception as e:
        logging.error(f"Sembol yükleme hatası: {str(e)}")
        return ['BTC/USDT', 'ETH/USDT']

# Global Ayarlar
CONFIG = {
    'SYMBOLS': load_usdt_futures_symbols(),
    'running': False,
    'LEVERAGE': 10,
    'RISK_PER_TRADE': 0.02,
    'TIMEFRAMES': ['5m', '15m', '1h', '4h'],
    'LOOKBACK_WINDOW': 1000,
    'MODELS': {},
    'SCALER': StandardScaler(),
    'chat_ids': set(),
    'last_signals': {},
    'performance_history': {symbol: deque(maxlen=100) for symbol in load_usdt_futures_symbols()},
    'MIN_ACCURACY': 0.6
}

# 1. Gelişmiş Veri Yönetimi Sistemi
class DataManager:
    @staticmethod
    async def fetch_and_store(symbol: str, timeframe: str):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['ema9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema21'] = talib.EMA(df['close'], timeperiod=21)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            df.to_sql('market_data', conn, if_exists='append', index=False)
            logging.info(f"{symbol} {timeframe} verisi güncellendi")
            return df.dropna()
        except Exception as e:
            logging.error(f"Veri çekme hatası: {str(e)}")
            return None

# 2. Çoklu Zaman Dilimi Özellik Mühendisliği
def create_multiframe_features(symbol: str):
    try:
        features = []
        for timeframe in CONFIG['TIMEFRAMES']:
            df = pd.read_sql(f"SELECT * FROM market_data WHERE symbol='{symbol}' AND timeframe='{timeframe}' ORDER BY timestamp DESC LIMIT 500", conn)
            if len(df) < 100:
                continue
                
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_change'] = df['volume'].pct_change()
            
            for window in [5, 20, 50]:
                df[f'ma_{window}'] = df['close'].rolling(window).mean()
                df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
                
            features.append(df.add_prefix(f'{timeframe}_'))
            
        return pd.concat(features, axis=1).ffill().dropna()
    except Exception as e:
        logging.error(f"Özellik oluşturma hatası: {str(e)}")
        return None

# 3. Kendi Kendine Öğrenen AI Model Sistemi
class AIModels:
    @staticmethod
    def build_lstm_model(input_shape):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def build_rf_model():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced'))
        ])

    @staticmethod
    async def train_or_update_model(symbol: str, model_type: str, X, y):
        try:
            model_path = f'/data/models/{symbol.replace("/", "_")}_{model_type.lower()}.h5' if model_type == 'LSTM' else f'/data/models/{symbol.replace("/", "_")}_{model_type.lower()}.pkl'
            if os.path.exists(model_path):
                if model_type == 'LSTM':
                    model = tf.keras.models.load_model(model_path)
                    model.fit(X, y, epochs=5, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=2)])
                else:
                    model = joblib.load(model_path)
                    model.fit(X, y)
            else:
                if model_type == 'LSTM':
                    model = AIModels.build_lstm_model((X.shape[1], X.shape[2]))
                    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                else:
                    model = AIModels.build_rf_model()
                    model.fit(X, y)

            os.makedirs('/data/models', exist_ok=True)
            if model_type == 'LSTM':
                model.save(model_path)
            else:
                joblib.dump(model, model_path)
            logging.info(f"{symbol} {model_type} modeli güncellendi")
            return model
        except Exception as e:
            logging.error(f"{symbol} {model_type} eğitim hatası: {str(e)}")
            return None

# 4. Otomatik Model Güncelleme ve Performans Takibi
async def periodic_training():
    logging.info("Periyodik model eğitimi başlatılıyor...")
    for symbol in CONFIG['SYMBOLS'][:10]:
        try:
            X = create_multiframe_features(symbol)
            if X is None or len(X) < 60:
                continue
            y = (X.filter(regex='close').iloc[:,0].pct_change().shift(-1) > 0).astype(int)
            
            seq_length = 60
            X_lstm = np.array([X.values[i-seq_length:i] for i in range(seq_length, len(X))])
            y_lstm = y[seq_length:]
            
            lstm_model = await AIModels.train_or_update_model(symbol, 'LSTM', X_lstm, y_lstm)
            rf_model = await AIModels.train_or_update_model(symbol, 'RF', X, y)
            
            if lstm_model and rf_model:
                lstm_acc = lstm_model.evaluate(X_lstm, y_lstm, verbose=0)[1]
                rf_acc = rf_model.score(X, y)
                c.execute("INSERT INTO model_performance VALUES (?, ?, ?, ?, ?)",
                         (symbol, 'LSTM', datetime.now(), lstm_acc, 0.0))
                c.execute("INSERT INTO model_performance VALUES (?, ?, ?, ?, ?)",
                         (symbol, 'RF', datetime.now(), rf_acc, 0.0))
                conn.commit()
        except Exception as e:
            logging.error(f"{symbol} model güncelleme hatası: {str(e)}")

# 5. Gelişmiş Sinyal Doğrulama
async def validate_signal(symbol: str, direction: str) -> bool:
    try:
        X = create_multiframe_features(symbol)
        if X is None or len(X) < 60:
            return False
        latest_data = X.iloc[-1:].values.reshape(1, -1)
        lstm_input = X.iloc[-60:].values.reshape(1, 60, -1)
        
        perf_df = pd.read_sql(f"SELECT * FROM model_performance WHERE symbol='{symbol}' ORDER BY timestamp DESC LIMIT 2", conn)
        if not perf_df.empty and perf_df['accuracy'].mean() < CONFIG['MIN_ACCURACY']:
            logging.info(f"{symbol} modelleri düşük doğruluk nedeniyle reddedildi")
            return False
        
        lstm_model = tf.keras.models.load_model(f'/data/models/{symbol.replace("/", "_")}_lstm.h5')
        lstm_prob = lstm_model.predict(lstm_input, verbose=0)[0][0]
        
        rf_model = joblib.load(f'/data/models/{symbol.replace("/", "_")}_rf.pkl')
        rf_prob = rf_model.predict_proba(latest_data)[0][1]
        
        combined_prob = (0.6 * lstm_prob + 0.4 * rf_prob)
        return combined_prob > 0.65
    except Exception as e:
        logging.error(f"Doğrulama hatası: {str(e)}")
        return False

# 6. Güncellenmiş Sinyal Üretim Sistemi
async def generate_ai_signal(symbol: str):
    if not CONFIG['running']:
        return
    
    try:
        await DataManager.fetch_and_store(symbol, '5m')
        await DataManager.fetch_and_store(symbol, '1h')
        
        df_5m = pd.read_sql(f"SELECT * FROM market_data WHERE symbol='{symbol}' AND timeframe='5m' ORDER BY timestamp DESC LIMIT 100", conn)
        df_1h = pd.read_sql(f"SELECT * FROM market_data WHERE symbol='{symbol}' AND timeframe='1h' ORDER BY timestamp DESC LIMIT 100", conn)
        
        long_signal = (df_5m['ema9'].iloc[-2] < df_5m['ema21'].iloc[-2] and
                       df_5m['ema9'].iloc[-1] > df_5m['ema21'].iloc[-1] and
                       df_1h['rsi'].iloc[-1] < 65)
        
        short_signal = (df_5m['ema9'].iloc[-2] > df_5m['ema21'].iloc[-2] and
                        df_5m['ema9'].iloc[-1] < df_5m['ema21'].iloc[-1] and
                        df_1h['rsi'].iloc[-1] > 35)
        
        if not (long_signal or short_signal):
            return
            
        direction = 'LONG' if long_signal else 'SHORT'
        
        if not await validate_signal(symbol, direction):
            logging.info(f"{symbol} AI tarafından onaylanmadı")
            return
            
        balance = exchange.fetch_balance()['USDT']['free']
        atr = df_5m['atr'].iloc[-1]
        entry_price = df_5m['close'].iloc[-1]
        risk_amount = balance * CONFIG['RISK_PER_TRADE']
        position_size = risk_amount / (atr * 1.5)
        
        signal = {
            'symbol': symbol,
            'direction': direction,
            'entry': entry_price,
            'tp': entry_price + (2 * atr) if direction == 'LONG' else entry_price - (2 * atr),
            'sl': entry_price - atr if direction == 'LONG' else entry_price + atr,
            'size': position_size,
            'timestamp': datetime.now().isoformat()
        }
        
        await broadcast_signal(signal)
        logging.info(f"AI Onaylı Sinyal: {signal}")
        
    except Exception as e:
        logging.error(f"Sinyal üretim hatası: {str(e)}")

# 7. Telegram Komutları ve Yayın
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    CONFIG['chat_ids'].add(chat_id)
    CONFIG['running'] = True
    await update.message.reply_text("Bot sinyal üretimine başladı! Durdurmak için /stop kullanabilirsiniz.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG['running'] = False
    await update.message.reply_text("Sinyal üretimi durduruldu. Yeniden başlatmak için /start kullanabilirsiniz.")

async def show_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CONFIG['last_signals']:
        await update.message.reply_text("Henüz sinyal yok.")
    else:
        signals = "\n".join([str(signal) for signal in CONFIG['last_signals'].values()])
        await update.message.reply_text(f"Mevcut Sinyaller:\n{signals}")

async def broadcast_signal(signal):
    CONFIG['last_signals'][signal['symbol']] = signal
    for chat_id in CONFIG['chat_ids']:
        await application.bot.send_message(chat_id, str(signal))

async def run_signal_generation():
    while True:
        if CONFIG['running']:
            for symbol in CONFIG['SYMBOLS']:
                await generate_ai_signal(symbol)
        await asyncio.sleep(300)  # 5 dakikada bir

# 8. Ana Çalışma
async def main():
    os.makedirs('/data/models', exist_ok=True)
    await periodic_training()

    global application
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("signals", show_signals))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(periodic_training, 'interval', hours=12)
    scheduler.start()

    logging.info("Bot başlatılıyor...")

    asyncio.create_task(run_signal_generation())
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
