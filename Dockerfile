FROM python:3.8-slim

WORKDIR /app

# Gerekli araçları yükle (git dahil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    cmake \
    libc6-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# GitHub'dan algdn46 deposunu klonla ve talib klasörünü kullan
RUN git clone https://github.com/Algdn46/algdn46.git \
    && mv algdn46/talib/* . \
    && rm -rf algdn46

# ta-lib Python paketini kur
RUN pip install --no-cache-dir ta-lib \
    || (wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib \
    && pip install --no-cache-dir ta-lib)

# requirements.txt ile bağımlılıkları kur
RUN pip install --no-cache-dir -r requirements.txt

# Uygulamayı çalıştır
CMD ["python", "bot.py"]
