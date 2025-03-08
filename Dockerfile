# Python 3.11 slim imajını temel al
FROM python:3.11-slim

# Sistem bağımlılıklarını kur (TA-Lib ve Python geliştirme araçları için)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    make \
    python3-dev \
    build-essential \
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

# Ortam değişkenlerini ayarla (TA-Lib'in bulunması için)
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# NumPy ve TensorFlow'u uyumlu sürümlerle kur
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir tensorflow-cpu==2.12.0

# TA-Lib Python paketini manuel olarak kur (requirements.txt'den önce)
RUN pip install --no-cache-dir TA-Lib==0.4.24

# Diğer bağımlılıkları requirements.txt dosyasından kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Uygulamayı çalıştır
CMD ["python", "bot.py"] 
