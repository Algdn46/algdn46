FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli araçları ve ta-lib'i kaynaktan derle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    make \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib \
    && pip install --no-cache-dir ta-lib \
    && apt-get purge --auto-remove -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Uygulama dosyalarını kopyala ve bağımlılıkları kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Uygulamayı çalıştır
CMD ["python", "main.py"]
