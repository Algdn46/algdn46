FROM python:3.11-slim

# Derleme araçlarını ve bağımlılıkları yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    make \
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

# NumPy'yi eski bir sürümle kur
RUN pip install numpy==1.26.4

# Uygulama bağımlılıklarını kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .
CMD ["python", "main.py"]
