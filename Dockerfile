FROM python:3.9-slim

# Çalışma dizini
WORKDIR /app

# Gerekli sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    build-essential \
    wget \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib kütüphanesini indir ve yükle
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Pip güncelle ve NumPy yükle
RUN pip install --upgrade pip
RUN pip install numpy==1.26.4

# TA-Lib Python bağlamasını yükle
RUN pip install TA-Lib==0.4.28

# Requirements.txt dosyasını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları konteynere kopyala
COPY . .

# Python çıktısını buffer'lamayı kapat
ENV PYTHONUNBUFFERED=1

# Konteyner başlatma komutu
CMD ["python", "bot.py"]
