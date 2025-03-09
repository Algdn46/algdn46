FROM python:3.8-slim

WORKDIR /app

# Gerekli araçları ve bağımlılıkları yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    cmake \
    libc6-dev \
    python3-dev \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib \
    && apt-get purge --auto-remove -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ta-lib Python paketini kur
RUN pip install --no-cache-dir ta-lib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
