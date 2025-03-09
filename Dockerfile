# Temel imaj olarak Python 3.10 kullanıyoruz
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını kur (TA-Lib ve diğer gerekli araçlar)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C kütüphanesini indir ve kur
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Gereksinim dosyasını kopyala
COPY requirements.txt .

# Python bağımlılıklarını kur
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Veri ve model dosyaları için dizin oluştur
RUN mkdir -p /data/models

# Ortam değişkenlerini ayarla (isteğe bağlı)
ENV PYTHONUNBUFFERED=1

# Uygulamayı çalıştır
CMD ["python", "main.py"]
