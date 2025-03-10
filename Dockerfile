# Base image
# Temel Python imajını kullan
FROM python:3.7.3-slim-stretch

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

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

# Copy application files
COPY . .

# Set entrypoint
CMD ["python", "bot.py"]
