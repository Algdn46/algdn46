FROM python:3.9-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle (TA-Lib için gerekli)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib'i derle ve kur
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Pip'i güncelle
RUN pip install --upgrade pip

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
