import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
import os
from dotenv import load_dotenv  # .env dosyasını okumak için
import signal
import sys
# .env dosyasını yükle
load_dotenv('gateio.env')

# ---------------------- API VE GLOBAL AYARLAR ----------------------
exchange = ccxt.gate({
    'apiKey': os.getenv('GATEIO_API_KEY'),
    'secret': os.getenv('GATEIO_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # Gate.io'da vadeli işlemler için genellikle 'swap' kullanılır
    },
})

# Global ayarlar
SYMBOLS = []  # Vadeli semboller dinamik olarak yüklenecek
TIMEFRAMES = ['5m', '15m', '1h']
LEVERAGE = 10
RISK_PER_TRADE = 0.02  # İşlem başına risk %2
LSTM_LOOKBACK = 50
EMA_PERIODS = (9, 21)
RSI_PERIOD = 14
ATR_PERIOD = 14
TRAIN_MODEL = False  # Eğer True ise eğitim fonksiyonu çalışır

open_positions = {}
signals = {}
lstm_model = None  # Global LSTM modeli

# API anahtarlarının varlığını kontrol et
if not os.getenv('GATEIO_API_KEY') or not os.getenv('GATEIO_SECRET_KEY'):
    raise ValueError("GATEIO_API_KEY veya GATEIO_SECRET_KEY .env dosyasında bulunamadı!")

# ---------------------- LSTM MODEL OLUŞTURMA & EĞİTİM ----------------------
def create_lstm_model(output_units=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, 3)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(output_units)  # Çıkış: Örneğin, sadece gelecek kapanış fiyatı
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model():
    try:
        data = pd.read_csv('historical_data.csv')
    except Exception as e:
        logging.error(f"Veri dosyası okunurken hata: {e}")
        return

    features = data[['close', 'high', 'low']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    def create_dataset(data_array, lookback):
        X, y = [], []
        for i in range(len(data_array) - lookback):
            X.append(data_array[i:i+lookback])
            y.append(data_array[i+lookback][0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_features, LSTM_LOOKBACK)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = create_lstm_model(output_units=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=[early_stop])
    model.save('lstm_model.h5')
    logging.info("LSTM modeli başarıyla eğitildi ve kaydedildi.")
    return model, scaler

def load_lstm_model():
    global lstm_model
    model_path = 'lstm_model.h5'
    if os.path.exists(model_path):
        try:
            lstm_model = load_model(model_path)
            logging.info("Önceden eğitilmiş LSTM modeli yüklendi.")
        except Exception as e:
            logging.error(f"LSTM modeli yüklenirken hata: {e}. Yeni model oluşturuluyor.")
            lstm_model = create_lstm_model()
    else:
        logging.info("LSTM modeli dosyası bulunamadı, yeni model oluşturuluyor.")
        lstm_model = create_lstm_model()

# ---------------------- DİNAMİK SEMBOL ÇEKİMİ ----------------------
def get_all_futures_symbols():
    try:
        markets = exchange.load_markets()
        logging.info(f"Toplam piyasa sayısı: {len(markets)}")
        futures_symbols = []
        for symbol, market in markets.items():
            if market.get('swap', False) and market.get('active', True):
                futures_symbols.append(symbol)
        logging.info(f"Bulunan vadeli semboller: {futures_symbols}")
        return futures_symbols
    except Exception as e:
        logging.error(f"Vadeli semboller alınırken hata: {str(e)}")
        return []

def update_symbols_periodically():
    global SYMBOLS
    while True:
        SYMBOLS = get_all_futures_symbols()
        logging.info(f"Güncel semboller: {SYMBOLS}")
        time.sleep(21600)  # 6 saatte bir güncelle

# ---------------------- VERİ TOPLAMA VE SİNYAL ÜRETİMİ ----------------------
def fetch_multi_tf_data(symbol):
    data = {}
    for tf in TIMEFRAMES:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['EMA9'] = df['close'].ewm(span=EMA_PERIODS[0]).mean()
            df['EMA21'] = df['close'].ewm(span=EMA_PERIODS[1]).mean()
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=RSI_PERIOD).mean()
            avg_loss = loss.rolling(window=RSI_PERIOD).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=ATR_PERIOD).mean()
            data[tf] = df
        except Exception as e:
            logging.error(f"{symbol} için {tf} zaman diliminde veri çekilirken hata: {e}")
    return data

def generate_signal(symbol):
    data = fetch_multi_tf_data(symbol)
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
    except Exception as e:
        logging.error(f"{symbol} için ticker alınırken hata: {e}")
        return

    try:
        trend_filter = (data['1h']['EMA9'].iloc[-1] > data['1h']['EMA21'].iloc[-1]) and \
                       (data['15m']['EMA9'].iloc[-1] > data['15m']['EMA21'].iloc[-1])
    except Exception as e:
        logging.error(f"{symbol} için trend filtresi hesaplanırken hata: {e}")
        trend_filter = False

    try:
        scaler = MinMaxScaler()
        recent_data = data['5m'][['close', 'high', 'low']].tail(LSTM_LOOKBACK)
        if len(recent_data) < LSTM_LOOKBACK:
            logging.warning(f"{symbol} için yeterli veri yok.")
            return
        scaled_data = scaler.fit_transform(recent_data)
        lstm_input = scaled_data.reshape(1, LSTM_LOOKBACK, 3)
        if lstm_model is None:
            load_lstm_model()
        prediction = lstm_model.predict(lstm_input)
        pred_price = prediction[0][0]
    except Exception as e:
        logging.error(f"{symbol} için LSTM tahmini yapılırken hata: {e}")
        pred_price = current_price

    try:
        long_condition = (data['5m']['EMA9'].iloc[-2] < data['5m']['EMA21'].iloc[-2] and
                          data['5m']['EMA9'].iloc[-1] > data['5m']['EMA21'].iloc[-1] and
                          data['5m']['RSI'].iloc[-1] < 65 and
                          trend_filter)
        short_condition = (data['5m']['EMA9'].iloc[-2] > data['5m']['EMA21'].iloc[-2] and
                           data['5m']['EMA9'].iloc[-1] < data['5m']['EMA21'].iloc[-1] and
                           data['5m']['RSI'].iloc[-1] > 35 and
                           trend_filter)
    except Exception as e:
        logging.error(f"{symbol} için sinyal koşulları belirlenirken hata: {e}")
        return

    try:
        atr = data['5m']['ATR'].iloc[-1]
        tp_long = current_price + 2 * atr
        sl_long = current_price - atr
        tp_short = current_price - 2 * atr
        sl_short = current_price + atr
    except Exception as e:
        logging.error(f"{symbol} için ATR hesaplanırken hata: {e}")
        return

    try:
        balance = exchange.fetch_balance()['USDT']['free']
        risk = abs(current_price - sl_long) if long_condition else abs(sl_short - current_price) if short_condition else 1
        position_size = (balance * RISK_PER_TRADE) / risk if risk != 0 else 0
    except Exception as e:
        logging.error(f"{symbol} için pozisyon boyutu hesaplanırken hata: {e}")
        position_size = 0

    signals[symbol] = {
        'symbol': symbol,
        'long': long_condition,
        'short': short_condition,
        'entry': current_price,
        'tp_long': tp_long,
        'sl_long': sl_long,
        'tp_short': tp_short,
        'sl_short': sl_short,
        'size': position_size,
        'time': datetime.now().strftime('%H:%M:%S')
    }

# ---------------------- POZİSYON YÖNETİMİ ----------------------
def open_position(signal):
    symbol = signal['symbol']
    try:
        exchange.set_leverage(LEVERAGE, symbol)
        if signal['long']:
            exchange.create_order(symbol, 'market', 'buy', signal['size'])
            open_positions[symbol] = {
                'side': 'long',
                'entry': signal['entry'],
                'tp': signal['tp_long'],
                'sl': signal['sl_long'],
                'size': signal['size']
            }
        elif signal['short']:
            exchange.create_order(symbol, 'market', 'sell', signal['size'])
            open_positions[symbol] = {
                'side': 'short',
                'entry': signal['entry'],
                'tp': signal['tp_short'],
                'sl': signal['sl_short'],
                'size': signal['size']
            }
        logging.info(f"{symbol} için pozisyon açıldı: {open_positions[symbol]}")
    except Exception as e:
        logging.error(f"{symbol} pozisyon açma hatası: {e}")

def close_position(symbol, position):
    try:
        if position['side'] == 'long':
            exchange.create_order(symbol, 'market', 'sell', position['size'])
        else:
            exchange.create_order(symbol, 'market', 'buy', position['size'])
        logging.info(f"{symbol} pozisyon kapatıldı.")
        del open_positions[symbol]
    except Exception as e:
        logging.error(f"{symbol} pozisyon kapatma hatası: {e}")

def manage_positions():
    while True:
        for symbol in list(open_positions.keys()):
            position = open_positions[symbol]
            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
            except Exception as e:
                logging.error(f"{symbol} için ticker alınırken hata: {e}")
                continue
            if position['side'] == 'long' and (current_price >= position['tp'] or current_price <= position['sl']):
                close_position(symbol, position)
            elif position['side'] == 'short' and (current_price <= position['tp'] or current_price >= position['sl']):
                close_position(symbol, position)
        time.sleep(10)

# ---------------------- GUI (Tkinter) ----------------------
class TradingGUI(tk.Tk):
    def _init_(self):
        super()._init_()
        self.title("Gelişmiş Gate.io Futures Trading Bot")
        self.geometry("1200x800")
        
        self.signal_frame = ttk.Frame(self)
        self.position_frame = ttk.Frame(self)
        
        self.signal_table = ttk.Treeview(self.signal_frame, columns=('Symbol', 'Direction', 'Entry', 'TP', 'SL', 'Size'))
        self.position_table = ttk.Treeview(self.position_frame, columns=('Symbol', 'Side', 'Entry', 'Current', 'TP', 'SL', 'PNL'))
        
        self._setup_gui()
        self.update_thread = threading.Thread(target=self.update_tables)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _setup_gui(self):
        for col in ('Symbol', 'Direction', 'Entry', 'TP', 'SL', 'Size'):
            self.signal_table.heading(col, text=col)
            self.signal_table.column(col, width=150)
        for col in ('Symbol', 'Side', 'Entry', 'Current', 'TP', 'SL', 'PNL'):
            self.position_table.heading(col, text=col)
            self.position_table.column(col, width=150)
        self.signal_frame.pack(fill='both', expand=True)
        self.position_frame.pack(fill='both', expand=True)
        self.signal_table.pack(fill='both', expand=True)
        self.position_table.pack(fill='both', expand=True)
    
    def update_tables(self):
        while True:
            try:
                self.signal_table.delete(*self.signal_table.get_children())
                for sym, sig in signals.items():
                    direction = 'LONG' if sig['long'] else 'SHORT' if sig['short'] else '-'
                    tp_value = sig['tp_long'] if direction == 'LONG' else sig['tp_short']
                    sl_value = sig['sl_long'] if direction == 'LONG' else sig['sl_short']
                    self.signal_table.insert('', 'end', values=(
                        sym, direction, f"{sig['entry']:.2f}", f"{tp_value:.2f}", f"{sl_value:.2f}", f"{sig['size']:.3f}"
                    ))
                self.position_table.delete(*self.position_table.get_children())
                for sym, pos in open_positions.items():
                    try:
                        ticker = exchange.fetch_ticker(sym)
                        current_price = ticker['last']
                        pnl = ((current_price - pos['entry']) / pos['entry']) * 100 if pos['side'] == 'long' \
                              else ((pos['entry'] - current_price) / pos['entry']) * 100
                        self.position_table.insert('', 'end', values=(
                            sym, pos['side'].upper(), f"{pos['entry']:.2f}",
                            f"{current_price:.2f}", f"{pos['tp']:.2f}",
                            f"{pos['sl']:.2f}", f"{pnl:.2f}%"
                        ))
                    except Exception as e:
                        logging.error(f"{sym} için GUI güncellemesi yapılırken hata: {e}")
                self.update_idletasks()
            except Exception as e:
                logging.error(f"GUI güncelleme hatası: {e}")
            time.sleep(5)

# ---------------------- ANA DÖNGÜ ----------------------
def main():
    logging.basicConfig(level=logging.DEBUG)
    # Test: Bakiyeyi çekmeyi dene
    try:
        balance = exchange.fetch_balance()
        logging.info(f"Bakiye: {balance['USDT']}")
    except Exception as e:
        logging.error(f"Bakiye çekme hatası: {str(e)}")

    # Ctrl+C için kapatma sinyali
    def signal_handler(sig, frame):
        logging.info("Bot kapatılıyor...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    if TRAIN_MODEL:
        model, scaler = train_lstm_model()
        global lstm_model
        lstm_model = model
    else:
        load_lstm_model()
    
    threading.Thread(target=update_symbols_periodically, daemon=True).start()
    
    gui = TradingGUI()
    
    def trading_loop():
        while True:
            try:
                if SYMBOLS:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        executor.map(generate_signal, SYMBOLS)
                    for sym, sig in signals.copy().items():
                        if (sig.get('long') or sig.get('short')) and sym not in open_positions:
                            open_position(sig)
                else:
                    logging.warning("Sembol listesi boş, tekrar kontrol ediliyor...")
            except Exception as e:
                logging.error(f"Trading loop hatası: {e}")
            time.sleep(60)
    
    threading.Thread(target=trading_loop, daemon=True).start()
    threading.Thread(target=manage_positions, daemon=True).start()
    gui.mainloop()
     
     
if __name__ == "__main__":
    main()