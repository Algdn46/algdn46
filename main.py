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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import os
import joblib
import talib
import tensorflow as tf
from collections import deque

# Sürüm Kontrolleri
assert tf.__version__ == '2.13.1', f"TensorFlow 2.13.1 gerekli! Mevcut: {tf.__version__}"
from tensorflow import __version__ as tf_version

# Async Çözümü
nest_asyncio.apply()

# Logging Konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_trading.log'),
        logging.StreamHandler()
    ]
)

# Çevresel Değişkenler
load_dotenv('config.env')

# Veritabanı Bağlantısı
def get_db_connection():
    conn = sqlite3.connect('trading_data.db', check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL;')  # Write-Ahead Logging için performans
    return conn
c = conn.cursor()

# Veritabanı Şeması
c.execute('''CREATE TABLE IF NOT EXISTS market_data
             (symbol TEXT, timeframe TEXT, timestamp DATETIME,
              open REAL, high REAL, low REAL, close REAL, volume REAL,
              ema9 REAL, ema21 REAL, rsi REAL, atr REAL,
              PRIMARY KEY (symbol, timeframe, timestamp))''')
conn.commit()
c.execute('''CREATE TABLE IF NOT EXISTS model_performance
             (symbol TEXT, model_type TEXT, timestamp DATETIME,
              accuracy REAL, profit REAL)''')
conn.commit()

# Exchange Konfigürasyonu
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# Sembol Yükleme Fonksiyonu
def load_usdt_futures_symbols():
    try:
        markets = exchange.load_markets()
        return [s for s in markets.keys() if s.endswith('/USDT')]
    except Exception as e:
        logging.error(f"Sembol yükleme hatası: {e}")
        return ['BTC/USDT', 'ETH/USDT']

# Global Konfigürasyon
CONFIG = {
 SYMBOLS = load_usdt_futures_symbols()
CONFIG['SYMBOLS'] = SYMBOLS      
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

# 1. Veri Yönetim Sistemi
class DataManager:
    @staticmethod
    async def fetch_and_store(symbol: str, timeframe: str):
        conn = get_db_connection()
        c = conn.cursor()
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['ema9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema21'] = talib.EMA(df['close'], timeperiod=21)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            for _, row in df.iterrows():
                c.execute('''INSERT OR REPLACE INTO market_data 
                             (symbol, timeframe, timestamp, open, high, low, close, volume, ema9, ema21, rsi, atr) 
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (symbol, timeframe, row['timestamp'], row['open'], row['high'], row['low'],
                           row['close'], row['volume'], row['ema9'], row['ema21'], row['rsi'], row['atr']))
            conn.commit()
            return df.dropna()
        except Exception as e:
            logging.error(f"Veri hatası ({symbol} {timeframe}): {str(e)}", exc_info=True)
            return None
        finally:
            conn.close()
           
# Teknik Göstergeleri İnceleme Fonksiyonu (Yeni Ek - Çakışma Yok)
def analyze_technical_indicators(df: pd.DataFrame) -> dict:
    """
    Teknik göstergeleri analiz eder ve durumu açıklar.
    - EMA Kesişimleri: Trend yönünü belirler.
    - RSI: Aşırı alım/satım durumlarını kontrol eder.
    - ATR: Volatilite ve risk seviyelerini değerlendirir.
    """
    analysis = {}
    
    # EMA Analizi
    if df['ema9'].iloc[-1] > df['ema21'].iloc[-1]:
        analysis['trend'] = 'Yükseliş (EMA9 > EMA21)'
    else:
        analysis['trend'] = 'Düşüş (EMA9 < EMA21)'
    
    # RSI Analizi
    rsi_latest = df['rsi'].iloc[-1]
    if rsi_latest > 70:
        analysis['rsi'] = f"Aşırı Alım ({rsi_latest:.2f})"
    elif rsi_latest < 30:
        analysis['rsi'] = f"Aşırı Satım ({rsi_latest:.2f})"
    else:
        analysis['rsi'] = f"Nötr ({rsi_latest:.2f})"
    
    # ATR Analizi
    atr_latest = df['atr'].iloc[-1]
    analysis['volatility'] = f"ATR: {atr_latest:.4f} (Yüksek volatilite riskli pozisyonlar için dikkat)"
    
    return analysis

# 2. Çoklu Zaman Dilimi Özellikleri
def create_multiframe_features(symbol: str):
    try:
        feature_frames = []
        for tf in CONFIG['TIMEFRAMES']:
            df = pd.read_sql(f"""
                SELECT * FROM market_data 
                WHERE symbol='{symbol}' AND timeframe='{tf}' 
                ORDER BY timestamp DESC LIMIT 500
            """, conn)
            
            if len(df) < 100:
                continue
                
            # Özellik Mühendisliği
            # - Teknik göstergeler burada ek özelliklerle zenginleştiriliyor
            # - Returns, volatilite ve MA oranları modele daha fazla bilgi sağlar
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_change'] = df['volume'].pct_change()
            
            for window in [5, 20, 50]:
                df[f'ma_{window}'] = df['close'].rolling(window).mean()
                df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
                
            feature_frames.append(df.add_prefix(f'{tf}_'))
            
        return pd.concat(feature_frames, axis=1).ffill().dropna()
    except Exception as e:
        logging.error(f"Özellik hatası ({symbol}): {e}")
        return None

# 3. AI Model Sistemi
class AIModels:
    @staticmethod
    def build_lstm(input_shape):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal'), input_shape=input_shape),
            Dropout(0.4),
            Bidirectional(LSTM(64, kernel_initializer='he_normal')),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision')]
        )
        return model

    @staticmethod
    def build_rf():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', warm_start=True))
        ])

    @staticmethod
    async def train_model(symbol: str, model_type: str, X, y):
        try:
            model_path = f"/data/models/{symbol.replace('/', '_')}_{model_type.lower()}.keras" if model_type == 'LSTM' else f"/data/models/{symbol.replace('/', '_')}_rf.pkl"
            
            if os.path.exists(model_path):
                if model_type == 'LSTM':
                    model = load_model(model_path)
                    for layer in model.layers[:-2]:
                        layer.trainable = False
                    model.fit(X, y, epochs=5, batch_size=64, verbose=0,
                              callbacks=[EarlyStopping(monitor='precision', patience=2, mode='max')])
                else:
                    model = joblib.load(model_path)
                    model.n_estimators += 50
                    model.fit(X, y)
            else:
                if model_type == 'LSTM':
                    model = AIModels.build_lstm((X.shape[1], X.shape[2]))
                    model.fit(X, y, epochs=50, batch_size=64, 
                             validation_split=0.2, verbose=0,
                             callbacks=[
                                 EarlyStopping(monitor='val_precision', patience=3, mode='max'),
                                 ModelCheckpoint(model_path, save_best_only=True)
                             ])
                else:
                    model = AIModels.build_rf()
                    model.fit(X, y)

            if model_type == 'LSTM':
                model.save(model_path, save_format='keras')
            else:
                joblib.dump(model, model_path)
                
            return model
        except Exception as e:
            logging.error(f"Model hatası ({symbol} {model_type}): {e}")
            return None

# 4. Model Güncelleme
async def update_models():
    logging.info("Model güncelleme başlatılıyor...")
    for symbol in CONFIG['SYMBOLS'][:10]:
        try:
            X = create_multiframe_features(symbol)
            if X is None or len(X) < 60:
                continue
                
            y = (X.filter(regex='close').iloc[:,0].pct_change().shift(-1) > 0).astype(int)
            
            seq_length = 60
            X_lstm = np.array([X.values[i-seq_length:i] for i in range(seq_length, len(X))])
            y_lstm = y[seq_length:]
            
            lstm_model = await AIModels.train_model(symbol, 'LSTM', X_lstm, y_lstm)
            rf_model = await AIModels.train_model(symbol, 'RF', X, y)
            
            if lstm_model and rf_model:
                lstm_acc = lstm_model.evaluate(X_lstm, y_lstm, verbose=0)[1]
                rf_acc = rf_model.score(X, y)
                c.execute("INSERT INTO model_performance VALUES (?, ?, ?, ?, ?)",
                         (symbol, 'LSTM', datetime.now(), lstm_acc, 0.0))
                c.execute("INSERT INTO model_performance VALUES (?, ?, ?, ?, ?)",
                         (symbol, 'RF', datetime.now(), rf_acc, 0.0))
                conn.commit()
        except Exception as e:
            logging.error(f"Güncelleme hatası ({symbol}): {e}")

# 5. Sinyal Doğrulama
async def validate_signal(symbol: str) -> bool:
    try:
        X = create_multiframe_features(symbol)
        if X is None or len(X) < 60:
            return False
            
        lstm_model = load_model(f'/data/models/{symbol.replace("/", "_")}_lstm.keras')
        rf_model = joblib.load(f'/data/models/{symbol.replace("/", "_")}_rf.pkl')
        
        lstm_input = X.iloc[-60:].values.reshape(1, 60, -1)
        lstm_prob = lstm_model.predict(lstm_input, verbose=0)[0][0]
        
        rf_input = X.iloc[-1:].values.reshape(1, -1)
        rf_prob = rf_model.predict_proba(rf_input)[0][1]
        
        # Teknik göstergeler burada dolaylı olarak modele etki ediyor (X üzerinden)
        return (0.6 * lstm_prob + 0.4 * rf_prob) > 0.65
    except Exception as e:
        logging.error(f"Doğrulama hatası ({symbol}): {e}")
        return False

# 6. Sinyal Üretim
  async def generate_signals():
    while True:
        if CONFIG['running']:
            logging.info("Sinyal üretimi başladı.")
            for i in range(0, len(CONFIG['SYMBOLS']), 5):  # 5'li gruplar
                batch = CONFIG['SYMBOLS'][i:i+5]
                for symbol in batch:
                    conn = get_db_connection()
                    try:
                        await DataManager.fetch_and_store(symbol, '5m')
                        await DataManager.fetch_and_store(symbol, '1h')
                    df_5m = pd.read_sql(
                        "SELECT * FROM market_data WHERE symbol=? AND timeframe=? ORDER BY timestamp DESC LIMIT 100",
                        conn, params=(symbol, '5m')
                    )
                    logging.info(f"{symbol} için {len(df_5m)} satır veri alındı.")
                    
                    long_signal = (
                        df_5m['ema9'].iloc[-2] < df_5m['ema21'].iloc[-2] and
                        df_5m['ema9'].iloc[-1] > df_5m['ema21'].iloc[-1] and
                        df_5m['rsi'].iloc[-1] < 65
                    )
                    short_signal = (
                        df_5m['ema9'].iloc[-2] > df_5m['ema21'].iloc[-2] and
                        df_5m['ema9'].iloc[-1] < df_5m['ema21'].iloc[-1] and
                        df_5m['rsi'].iloc[-1] > 35
                    )
                    logging.info(f"{symbol}: Long: {long_signal}, Short: {short_signal}")
                    
                    if not (long_signal or short_signal):
                        continue
                    
                    direction = 'LONG' if long_signal else 'SHORT'
                    if await validate_signal(symbol):
                        balance = exchange.fetch_balance()['USDT']['free']
                        atr = df_5m['atr'].iloc[-1]
                        entry = df_5m['close'].iloc[-1]
                        risk_amount = balance * CONFIG['RISK_PER_TRADE']
                        size = risk_amount / (atr * 1.5)
                        
                        signal = {
                            'symbol': symbol,
                            'direction': direction,
                            'entry': round(entry, 4),
                            'tp': round(entry + (2 * atr) if direction == 'LONG' else entry - (2 * atr), 4),
                            'sl': round(entry - atr if direction == 'LONG' else entry + atr, 4),
                            'size': round(size, 4),
                            'timestamp': datetime.now().isoformat()
                        }
                        await broadcast_signal(signal)
                        logging.info(f"Yeni Sinyal: {signal}")
                    else:
                        logging.info(f"{symbol}: Sinyal doğrulanmadı.")
          except Exception as e:
                        logging.error(f"Sinyal hatası ({symbol}): {str(e)}", exc_info=True)
                    finally:
                        conn.close()
                await asyncio.sleep(10)  # Grup arasında bekleme
        await asyncio.sleep(300)

# 7. Telegram Entegrasyonu
async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG['chat_ids'].add(update.message.chat_id)
    CONFIG['running'] = True
    await update.message.reply_text("Trading Bot aktif!")

async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG['running'] = False
    await update.message.reply_text("Trading Bot durduruldu.")

async def broadcast_signal(signal):
    for chat_id in CONFIG['chat_ids']:
        try:
            await application.bot.send_message(
                chat_id,
                f"🚨 {signal['symbol']} {signal['direction']}\n"
                f"Entry: {signal['entry']}\n"
                f"TP: {signal['tp']}\n"
                f"SL: {signal['sl']}\n"
                f"Boyut: {signal['size']}"
            )
        except Exception as e:
            logging.error(f"Yayın hatası ({chat_id}): {e}")

# 8. Ana Program
async def main():
    os.makedirs('/data/models', exist_ok=True)
    global application
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
    application.add_handler(CommandHandler("start", start_bot))
    application.add_handler(CommandHandler("stop", stop_bot))
    scheduler = AsyncIOScheduler()
    scheduler.add_job(update_models, 'interval', hours=12)
    scheduler.start()
    tasks = [asyncio.create_task(generate_signals())]
    await application.run_polling()
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program kapatıldı.")
