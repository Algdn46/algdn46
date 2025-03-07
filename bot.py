import ccxt
import pandas as pd
import numpy as np
import asyncio
import logging
import nest_asyncio
import sqlite3
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
import os
import joblib
import talib

# Async sorunları için
nest_asyncio.apply()

# Logging Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ai_trading.log'), logging.StreamHandler()]
)

# Çevresel Değişkenler
load_dotenv('gateio.env')

# Veritabanı Bağlantısı
conn = sqlite3.connect('trading_data.db')
c = conn.cursor()

# Veritabanı Tablosu Oluşturma
c.execute('''CREATE TABLE IF NOT EXISTS market_data
             (symbol TEXT, timeframe TEXT, timestamp DATETIME,
              open REAL, high REAL, low REAL, close REAL, volume REAL,
              ema9 REAL, ema21 REAL, rsi REAL, atr REAL, PRIMARY KEY (symbol, timeframe, timestamp))''')
conn.commit()

# Global Ayarlar
CONFIG = {
    'SYMBOLS': [],
    'running': False,
    'LEVERAGE': 10,
    'RISK_PER_TRADE': 0.02,
    'TIMEFRAMES': ['5m', '15m', '1h', '4h'],
    'LOOKBACK_WINDOW': 1000,
    'MODELS': {
        'LSTM': None,
        'RF': None,
        'ENSEMBLE_WEIGHTS': {'LSTM': 0.6, 'RF': 0.4}
    },
    'SCALER': StandardScaler(),
    'chat_ids': set(),
    'last_signals': {},
    'position_size': 0.0
}

# 1. Gelişmiş Veri Yönetimi Sistemi
class DataManager:
    @staticmethod
    async def fetch_and_store(symbol: str, timeframe: str):
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Teknik Göstergeler
            df['ema9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema21'] = talib.EMA(df['close'], timeperiod=21)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Veritabanına Kaydet
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
                
            # Hesaplanmış Özellikler
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_change'] = df['volume'].pct_change()
            
            # Zaman Serisi Özellikleri
            for window in [5, 20, 50]:
                df[f'ma_{window}'] = df['close'].rolling(window).mean()
                df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
                
            features.append(df.add_prefix(f'{timeframe}_'))
            
        return pd.concat(features, axis=1).ffill().dropna()
    except Exception as e:
        logging.error(f"Özellik oluşturma hatası: {str(e)}")
        return None

# 3. Ensemble AI Model Sistemi
class AIModels:
    @staticmethod
    def train_lstm_model(X, y):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), 
            Dropout(0.4),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        return model

    @staticmethod
    def train_random_forest(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced'))
        ])
        
        param_dist = {
            'clf__max_depth': randint(3, 10),
            'clf__min_samples_split': randint(2, 10)
        }
        
        search = RandomizedSearchCV(pipeline, param_dist, n_iter=5, cv=3, scoring='accuracy')
        search.fit(X, y)
        return search.best_estimator_

# 4. Otomatik Model Güncelleme Sistemi
async def periodic_training():
    logging.info("Periyodik model eğitimi başlatılıyor...")
    symbols = pd.read_sql("SELECT DISTINCT symbol FROM market_data", conn)['symbol'].tolist()
    
    for symbol in symbols[:10]:  # İlk 10 sembol için
        try:
            X = create_multiframe_features(symbol)
            y = (X.filter(regex='close').iloc[:,0].pct_change().shift(-1) > 0).astype(int)
            
            # LSTM için veri hazırlığı
            seq_length = 60
            X_lstm = np.array([X.values[i-seq_length:i] for i in range(seq_length, len(X))])
            y_lstm = y[seq_length:]
            
            # Model Eğitimi
            lstm_model = AIModels.train_lstm_model(X_lstm, y_lstm)
            rf_model = AIModels.train_random_forest(X, y)
            
            # Modelleri Kaydet
            lstm_model.save(f'models/{symbol}_lstm.h5')
            joblib.dump(rf_model, f'models/{symbol}_rf.pkl')
            
            logging.info(f"{symbol} modelleri güncellendi")
        except Exception as e:
            logging.error(f"{symbol} model güncelleme hatası: {str(e)}")

# 5. Gelişmiş Sinyal Doğrulama
async def validate_signal(symbol: str, direction: str) -> bool:
    try:
        # 1. Tarihsel Başarı Oranı Kontrolü
        historical = pd.read_sql(f"""SELECT direction, outcome FROM signals 
                                   WHERE symbol='{symbol}' AND direction='{direction}'
                                   ORDER BY timestamp DESC LIMIT 100""", conn)
        if len(historical) > 20 and historical['outcome'].mean() < 0.55:
            return False
            
        # 2. Ensemble Model Tahmini
        X = create_multiframe_features(symbol)
        latest_data = X.iloc[-1:].values.reshape(1, -1)
        
        # LSTM Tahmini
        lstm_model = tf.keras.models.load_model(f'models/{symbol}_lstm.h5')
        lstm_input = X.iloc[-60:].values.reshape(1, 60, -1)
        lstm_prob = lstm_model.predict(lstm_input)[0][0]
        
        # RF Tahmini
        rf_model = joblib.load(f'models/{symbol}_rf.pkl')
        rf_prob = rf_model.predict_proba(latest_data)[0][1]
        
        # Kombine Tahmin
        combined_prob = (CONFIG['MODELS']['ENSEMBLE_WEIGHTS']['LSTM'] * lstm_prob +
                        CONFIG['MODELS']['ENSEMBLE_WEIGHTS']['RF'] * rf_prob)
                        
        return combined_prob > 0.65
    except Exception as e:
        logging.error(f"Doğrulama hatası: {str(e)}")
        return False

# 6. Güncellenmiş Sinyal Üretim Sistemi
async def generate_ai_signal(symbol: str):
    try:
        # Verileri Güncelle
        await DataManager.fetch_and_store(symbol, '5m')
        await DataManager.fetch_and_store(symbol, '1h')
        
        # Temel Sinyal Üretimi
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
        
        # AI Doğrulaması
        if not await validate_signal(symbol, direction):
            logging.info(f"{symbol} AI tarafından onaylanmadı")
            return
            
        # Risk Yönetimi
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
            'confidence': combined_prob,
            'timestamp': datetime.now().isoformat()
        }
        
        await broadcast_signal(signal)
        logging.info(f"AI Onaylı Sinyal: {signal}")
        
    except Exception as e:
        logging.error(f"Sinyal üretim hatası: {str(e)}")

# 7. Render için Gerekli Ayarlar
if __name__ == '__main__':
    # Model klasörünü oluştur
    os.makedirs('models', exist_ok=True)
    
    # Başlangıçta model eğitimi
    asyncio.run(periodic_training())
    
    # Zamanlayıcıları Ayarla
    scheduler = AsyncIOScheduler()
    scheduler.add_job(periodic_training, 'interval', hours=12)
    scheduler.add_job(run_signal_generation, 'interval', minutes=5)
    scheduler.start()
    
    # Telegram Botu Başlat
    application = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("signals", show_signals))
    application.run_polling()
