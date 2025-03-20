# AI Trading Bot
Binance Futures için yapay zeka destekli bir işlem botu. LSTM ve Random Forest modelleriyle sinyal üretir ve Telegram üzerinden bildirim gönderir.

## Özellikler
- Çoklu zaman dilimi analizi (5m, 15m, 1h, 4h)
- Teknik göstergeler: EMA, RSI, ATR (TA-Lib ile)
- Yapay zeka ile sinyal doğrulama (LSTM ve Random Forest)
- Telegram entegrasyonu

## Kurulum
1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/<username>/ai-trading-bot.git
   cd ai-trading-bot
