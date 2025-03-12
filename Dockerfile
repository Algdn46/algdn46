FROM python:3.11-slim
# Pip'i güncelle
RUN pip install --upgrade pip
# TA-Lib için ek kaynak ekle ve bağımlılıkları kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get install -y --no-install-recommends \
       gcc g++ libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# TA-Lib Python paketini kur
RUN pip install ta-lib
# Çalışma dizinini ayarla
WORKDIR /app
# Proje bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "bot.py"]

