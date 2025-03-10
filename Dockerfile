
# Temel Python imajını kullan
FROM python:3.11-slim
RUN pip install --upgrade pip

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pip'i güncelle


# Python bağımlılıklarını yüklemek için requirements.txt oluştur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY bot.py .
COPY andirin.env /app/andirin.env

# Veri ve model dosyaları için kalıcı bir volume dizini oluştur
RUN mkdir -p /data/models

# Volume tanımla (veritabanı ve modeller için)
VOLUME ["/data"]

# Ortam değişkenlerini ayarla
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Uygulamayı çalıştır
CMD ["python", "bot.py"]
