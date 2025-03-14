import ccxt
import openai
import sqlite3
import telebot
import time
import os
import logging
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

# Logging konfigurasyonu
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API anahtarlarını yükle
load_dotenv("config.env")

# Gerekli çevresel değişkenleri kontrol et
required_envs = {
    "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
    "BINANCE_SECRET_KEY": os.getenv("BINANCE_SECRET_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID")
}

for env_name, env_value in required_envs.items():
    if not env_value:
        raise ValueError(f"{env_name} çevresel değişkeni bulunamadı!")

# Binance Futures bağlantısı
exchange = ccxt.binance({
    "apiKey": required_envs["BINANCE_API_KEY"],
    "secret": required_envs["BINANCE_SECRET_KEY"],
    "options": {"defaultType": "future"},
    "timeout": 30000  # 30 saniye timeout
})

# OpenAI API
openai.api_key = required_envs["OPENAI_API_KEY"]

# Telegram bot
bot = telebot.TeleBot(required_envs["TELEGRAM_BOT_TOKEN"])
TELEGRAM_CHAT_ID = required_envs["TELEGRAM_CHAT_ID"]

class DatabaseManager:
    def _init_(self, db_name="trading_data.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                symbol TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, entry, sl, tp)
            )
        """)
        self.conn.commit()

    def already_sent(self, symbol, entry, sl, tp):
        self.cursor.execute(
            "SELECT * FROM signals WHERE symbol=? AND entry=? AND sl=? AND tp=?",
            (symbol, entry, sl, tp)
        )
        return self.cursor.fetchone() is not None

    def save_signal(self, symbol, entry, sl, tp):
        self.cursor.execute(
            "INSERT OR REPLACE INTO signals (symbol, entry, sl, tp) VALUES (?, ?, ?, ?)",
            (symbol, entry, sl, tp)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

@sleep_and_retry
@limits(calls=10, period=60)  # Dakikada 10 istek
def fetch_ticker_with_limit(exchange, symbol):
    return exchange.fetch_ticker(symbol)

def generate_signal(symbol, price):
    """Yapay zeka ile sinyal üretir"""
    prompt = f"{symbol} için Binance Futures analiz yap. Güncel fiyat {price} USDT. Long mu Short mu almalıyım? SL, TP ve risk hesapla."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "Sadece kısa ve net bir sinyal ver."},
                     {"role": "user", "content": prompt}]
        )
        signal_text = response["choices"][0]["message"]["content"]
        return parse_signal(signal_text)
    except Exception as e:
        logging.error(f"OpenAI sinyal üretimi hatası: {e}")
        return None

def parse_signal(text):
    """Sinyali ayrıştırır"""
    try:
        lines = text.strip().split("\n")
        if not lines or len(lines) < 5:
            return None
            
        direction = "Long" if "Long" in lines[0].upper() else "Short"
        
        def extract_value(line):
            return float(line.split(":")[1].strip().replace("USDT", "").strip())
            
        entry = extract_value(lines[1])
        sl = extract_value(lines[2])
        tp = extract_value(lines[3])
        risk = extract_value(lines[4])
        
        # Mantıksal kontrol
        if direction == "Long" and (sl >= entry or tp <= entry):
            return None
        if direction == "Short" and (sl <= entry or tp >= entry):
            return None
            
        return direction, entry, sl, tp, risk
    except (IndexError, ValueError) as e:
        logging.error(f"Sinyal ayrıştırma hatası: {e}\nText: {text}")
        return None

def send_telegram_message(symbol, direction, entry, sl, tp, risk):
    """Telegram'a sinyal gönderir"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    message = f"""
📢 *{symbol} {direction} Sinyali* ({timestamp})
📌 Giriş: {entry} USDT
📉 Stop-Loss: {sl} USDT
📈 Take-Profit: {tp} USDT
⚠️ Risk: {risk}
    """
    try:
        bot.send_message(TELEGRAM_CHAT_ID, message, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Telegram mesaj gönderme hatası: {e}")

def main():
    db = DatabaseManager()
    try:
        while True:
            try:
                markets = exchange.load_markets()
                symbols = [s for s in markets if "/USDT" in s and "PERP" in s]

                for symbol in symbols:
                    try:
                        ticker = fetch_ticker_with_limit(exchange, symbol)
                        price = ticker["last"]
                        
                        ai_signal = generate_signal(symbol, price)
                        
                        if ai_signal:
                            direction, entry, sl, tp, risk = ai_signal
                            
                            if not db.already_sent(symbol, entry, sl, tp):
                                send_telegram_message(symbol, direction, entry, sl, tp, risk)
                                db.save_signal(symbol, entry, sl, tp)
                                logging.info(f"Sinyal gönderildi: {symbol} {direction}")
                                
                    except Exception as e:
                        logging.error(f"Sembol {symbol} için hata: {e}")
                        
                time.sleep(60 * 5)  # 5 dakika bekle
                
            except Exception as e:
                logging.error(f"Ana döngüde hata: {e}")
                time.sleep(60)
                
    finally:
        db.close()

if _name_ == "_main_":
    main()

