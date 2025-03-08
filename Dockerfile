# Python 3.11 tabanlı Docker imajı
FROM python:3.11

# Gerekli sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C kütüphanesini kur
RUN curl -L -o ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j$(nproc) || make || make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Kütüphane yollarını güncelle
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# TA-Lib'in çalıştığını kontrol et
RUN ldconfig -p | grep ta_lib

# TA-Lib Python paketini yükle
RUN pip install --no-cache-dir TA-Lib==0.6.3

# Diğer bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Model dosyaları için dizin oluştur
RUN mkdir -p /data/models

# Uygulamayı başlat
CMD ["python", "bot.py"]

