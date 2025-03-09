FROM python:3.8-slim

WORKDIR /app

# Gerekli araçları yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python-dotenv'u kur (andirin.env dosyasını okumak için)
RUN pip install --no-cache-dir python-dotenv

# Kök dizinden andirin.env dosyasını kopyala ve GITHUB_TOKEN ile depoyu klonla
COPY andirin.env .
RUN export $(grep -v '^#' andirin.env | xargs) \
    && git clone https://${GITHUB_TOKEN}@github.com/Algdn46/algdn46.git \
    && mv algdn46/talib/* . \
    && mv algdn46/bot.py . \
    && mv algdn46/requirements.txt . \
    && mv algdn46/andirin.env . \
    && rm -rf algdn46

# talib-binary Python paketini kur
RUN pip install --no-cache-dir talib-binary==0.4.28

# requirements.txt ile bağımlılıkları kur
RUN pip install --no-cache-dir -r requirements.txt

# Uygulamayı çalıştır
CMD ["python", "bot.py"]
