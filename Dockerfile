FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib kaynak kodunu indir ve derle
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install

# Proje dosyalarını kopyala ve bağımlılıkları kur
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
