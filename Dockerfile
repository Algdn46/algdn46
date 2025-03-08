FROM python:3.11

# Sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    make \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C kütüphanesini kur
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# Bağımlılıkları kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Kalıcı disk için dizin oluştur
RUN mkdir -p /data/models

# Uygulamayı çalıştır
CMD ["python", "bot.py"]
