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
import os
from dotenv import load_dotenv
import telegram
from telegram.ext import Updater, CommandHandler

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Repository'den dosyaların bulunduğu dizini belirt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # bot.py dosyasının bulunduğu dizin
ENV_FILE = os.path.join(BASE_DIR, 'gateio.env')

# .env dosyasını yükle
if not os.path.exists(ENV_FILE):
    raise FileNotFoundError(f"{ENV_FILE} dosyası bulunamadı!")
load_dotenv(ENV_FILE)

# Gate.io API
exchange = ccxt.gate({
    'apiKey': os.getenv('GATEIO_API_KEY'),
    'secret': os.getenv('GATEIO_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

# Telegram Bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN or not os.getenv('GATEIO_API_KEY') or not os.getenv('GATEIO_SECRET_KEY'):
    raise ValueError("API anahtarları veya Telegram token eksik!")
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Global ayarlar
SYMBOLS = []
running = True
TIMEFRAMES = ['5m', '15m', '1h']
LEVERAGE = 10
RISK_PER_TRADE = 0.02
LSTM_LOOKBACK = 50
EMA_PERIODS = (9, 21)
RSI_PERIOD = 14
ATR_PERIOD = 14
chat_ids = set()
signals = {}
open_positions = {}
lstm_model = None
signals_lock = threading.Lock()
positions_lock = threading.Lock()

# LSTM Model
def create_lstm_model(output_units=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, 3)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(output_units)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def load_lstm_model():
    global lstm_model
    model_path = os.path.join(BASE_DIR, 'lstm_model.h5')  # Model dosyasını BASE_DIR'den ara
    if os.path.exists(model_path):
        try:
            lstm_model = load_model(model_path)
            logging.info("LSTM modeli yüklendi.")
        except Exception as e:
            logging.error(f"LSTM modeli yükleme hatası: {e}")
            lstm_model = create_lstm_model()
    else:
        lstm_model = create_lstm_model()
        logging.info("Yeni LSTM modeli oluşturuldu.")

# Sembol Çekme
def get_all_futures_symbols():
    try:
        markets = exchange.load_markets()
        logging.info(f"Toplam piyasa: {len(markets)}")
        futures_symbols = [symbol for symbol, market in markets.items() if market.get('type') == 'swap' and market.get('active', True)]
        logging.info(f"Vadeli semboller: {len(futures_symbols)} adet")
        return futures_symbols
    except ccxt.NetworkError as e:
        logging.error(f"Sembol çekme ağ hatası: {e}")
        return []
    except ccxt.ExchangeError as e:
        logging.error(f"Sembol çekme borsa hatası: {e}")
        return []
    except Exception as e:
        logging.error(f"Sembol çekme bilinmeyen hata: {e}")
        return []

# Veri Toplama ve Sinyal Üretimi
def fetch_multi_tf_data(symbol):
    data = {}
    for tf in TIMEFRAMES:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['EMA9'] = df['close'].ewm(span=EMA_PERIODS[0]).mean()
            df['EMA21'] = df['close'].ewm(span=EMA_PERIODS[1]).mean()
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            df['RSI'] = 100 - (100 / (1 + gain.rolling(window=RSI_PERIOD).mean() / loss.rolling(window=RSI_PERIOD).mean()))
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=ATR_PERIOD).mean()
            data[tf] = df
        except ccxt.NetworkError as e:
            logging.error(f"{symbol} {tf} ağ hatası: {e}")
        except ccxt.ExchangeError as e:
            logging.error(f"{symbol} {tf} borsa hatası: {e}")
        except Exception as e:
            logging.error(f"{symbol} {tf} bilinmeyen hata: {e}")
    return data

def generate_signal(symbol):
    data = fetch_multi_tf_data(symbol)
    if not data or '5m' not in data or len(data['5m']) < LSTM_LOOKBACK:
        return
    try:
        current_price = exchange.fetch_ticker(symbol)['last']
        trend_filter = (data['1h']['EMA9'].iloc[-1] > data['1h']['EMA21'].iloc[-1]) and (data['15m']['EMA9'].iloc[-1] > data['15m']['EMA21'].iloc[-1])
        scaler = MinMaxScaler()
        recent_data = data['5m'][['close', 'high', 'low']].tail(LSTM_LOOKBACK)
        scaled_data = scaler.fit_transform(recent_data)
        lstm_input = scaled_data.reshape(1, LSTM_LOOKBACK, 3)
        if lstm_model is None:
            load_lstm_model()
        if lstm_model is None:
            logging.error(f"{symbol} için LSTM modeli yüklenemedi.")
            return
        pred_price = lstm_model.predict(lstm_input, verbose=0)[0][0]
        long_condition = (data['5m']['EMA9'].iloc[-2] < data['5m']['EMA21'].iloc[-2] and
                          data['5m']['EMA9'].iloc[-1] > data['5m']['EMA21'].iloc[-1] and
                          data['5m']['RSI'].iloc[-1] < 65 and trend_filter)
        short_condition = (data['5m']['EMA9'].iloc[-2] > data['5m']['EMA21'].iloc[-2] and
                           data['5m']['EMA9'].iloc[-1] < data['5m']['EMA21'].iloc[-1] and
                           data['5m']['RSI'].iloc[-1] > 35 and trend_filter)
        if not (long_condition or short_condition):
            return
        atr = data['5m']['ATR'].iloc[-1]
        tp_long, sl_long = current_price + 2 * atr, current_price - atr
        tp_short, sl_short = current_price - 2 * atr, current_price + atr
        balance = exchange.fetch_balance()['USDT']['free']
        risk = abs(current_price - sl_long) if long_condition else abs(sl_short - current_price)
        position_size = (balance * RISK_PER_TRADE) / risk if risk != 0 else 0

        with signals_lock:
            signals[symbol] = {
                'symbol': symbol, 'long': long_condition, 'short': short_condition,
                'entry': current_price, 'tp_long': tp_long, 'sl_long': sl_long,
                'tp_short': tp_short, 'sl_short': sl_short, 'size': position_size,
                'time': datetime.now().strftime('%H:%M:%S')
            }
        if chat_ids:
            direction = 'LONG' if long_condition else 'SHORT'
            message = f"📈 Sinyal\nSembol: {symbol}\nYön: {direction}\nGiriş: {current_price:.2f}\nTP: {tp_long if direction == 'LONG' else tp_short:.2f}\nSL: {sl_long if direction == 'LONG' else sl_short:.2f}\nBoyut: {position_size:.3f}"
            for chat_id in chat_ids:
                bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        logging.error(f"{symbol} sinyal hatası: {e}")

# Pozisyon Yönetimi
def open_position(signal):
    symbol = signal['symbol']
    try:
        exchange.set_leverage(LEVERAGE, symbol)
        if signal['long']:
            exchange.create_order(symbol, 'market', 'buy', signal['size'])
            with positions_lock:
                open_positions[symbol] = {'side': 'long', 'entry': signal['entry'], 'tp': signal['tp_long'], 'sl': signal['sl_long'], 'size': signal['size']}
        elif signal['short']:
            exchange.create_order(symbol, 'market', 'sell', signal['size'])
            with positions_lock:
                open_positions[symbol] = {'side': 'short', 'entry': signal['entry'], 'tp': signal['tp_short'], 'sl': signal['sl_short'], 'size': signal['size']}
        if chat_ids:
            message = f"✅ {symbol} {open_positions[symbol]['side'].upper()} pozisyon açıldı!"
            for chat_id in chat_ids:
                bot.send_message(chat_id=chat_id, text=message)
    except ccxt.ExchangeError as e:
        logging.error(f"{symbol} pozisyon açma borsa hatası: {e}")
    except Exception as e:
        logging.error(f"{symbol} pozisyon açma hatası: {e}")

def close_position(symbol, position):
    try:
        if position['side'] == 'long':
            exchange.create_order(symbol, 'market', 'sell', position['size'])
        else:
            exchange.create_order(symbol, 'market', 'buy', position['size'])
        if chat_ids:
            message = f"❌ {symbol} pozisyon kapatıldı."
            for chat_id in chat_ids:
                bot.send_message(chat_id=chat_id, text=message)
        with positions_lock:
            del open_positions[symbol]
    except ccxt.ExchangeError as e:
        logging.error(f"{symbol} pozisyon kapatma borsa hatası: {e}")
    except Exception as e:
        logging.error(f"{symbol} pozisyon kapatma hatası: {e}")

# Telegram Komutları
def start(update, context):
    chat_id = update.message.chat_id
    chat_ids.add(chat_id)
    update.message.reply_text("Gate.io Trading Bot aktif! Sinyaller burada.\nKomutlar: /stop, /signals, /positions")

def stop(update, context):
    global running
    running = False
    update.message.reply_text("Bot durduruluyor...")
    time.sleep(2)
    context.job_queue.stop()

def show_signals(update, context):
    with signals_lock:
        if not signals:
            update.message.reply_text("Henüz sinyal yok.")
        else:
            message = "📈 Güncel Sinyaller\n"
            for sym, sig in signals.items():
                direction = 'LONG' if sig['long'] else 'SHORT' if sig['short'] else 'NONE'
                message += f"{sym}: {direction} - Giriş: {sig['entry']:.2f}\n"
            update.message.reply_text(message)

def show_positions(update, context):
    with positions_lock:
        if not open_positions:
            update.message.reply_text("Açık pozisyon yok.")
        else:
            message = "📊 Açık Pozisyonlar\n"
            for sym, pos in open_positions.items():
                message += f"{sym}: {pos['side'].upper()} - Giriş: {pos['entry']:.2f}\n"
            update.message.reply_text(message)

# Ana Döngü
def trading_loop():
    while running:
        try:
            if SYMBOLS:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    executor.map(generate_signal, SYMBOLS)
                with signals_lock:
                    for sym, sig in signals.copy().items():
                        if (sig.get('long') or sig.get('short')) and sym not in open_positions:
                            open_position(sig)
                with positions_lock:
                    for symbol in list(open_positions.keys()):
                        position = open_positions[symbol]
                        current_price = exchange.fetch_ticker(symbol)['last']
                        if position['side'] == 'long' and (current_price >= position['tp'] or current_price <= position['sl']):
                            close_position(symbol, position)
                        elif position['side'] == 'short' and (current_price <= position['tp'] or current_price >= position['sl']):
                            close_position(symbol, position)
            time.sleep(60)
        except ccxt.RequestTimeout:
            logging.error("Borsa isteği zaman aşımına uğradı, döngü devam ediyor.")
        except Exception as e:
            logging.error(f"Trading loop hatası: {e}")

def update_symbols_periodically():
    while running:
        global SYMBOLS
        new_symbols = get_all_futures_symbols()
        if new_symbols:
            SYMBOLS = new_symbols
        time.sleep(300)

def main():
    load_lstm_model()
    global SYMBOLS
    SYMBOLS = get_all_futures_symbols()
    if not SYMBOLS:
        logging.error("Sembol listesi boş, bot durduruluyor.")
        return
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(CommandHandler("signals", show_signals))
    dp.add_handler(CommandHandler("positions", show_positions))
    threading.Thread(target=update_symbols_periodically, daemon=True).start()
    threading.Thread(target=trading_loop, daemon=True).start()
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
