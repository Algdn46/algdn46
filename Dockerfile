# Python 3.11 imajını kullan
FROM python:3.11

# Gerekli bağımlılıkları yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    libatlas-base-dev \
    ta-lib \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib Python paketini yükle
RUN pip install --no-cache-dir TA-Lib==0.6.3

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Model dosyaları için dizin oluştur
RUN mkdir -p /data/models

# Uygulamayı başlat
CMD ["python", "bot.py"]
