# Python 3.11 imajını kullan
FROM python:3.11

# Gerekli sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    make \
    python3-dev \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C kütüphanesini indir ve kur
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# TA-Lib için ortam değişkeni ayarla
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Model dosyaları için dizin oluştur
RUN mkdir -p /data/models

# Uygulamayı başlat
CMD ["python", "bot.py"]
