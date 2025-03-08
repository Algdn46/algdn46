FROM python:3.10-slim

WORKDIR /app

# TA-Lib kurulumu
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install

# Gereksiz dosyaları temizle
RUN rm -rf ta-lib ta-lib-0.4.0-src.tar.gz


# Gerekli derleme araçlarını yükle
RUN apt-get update && apt-get install -y gcc g++ libffi-dev

# NumPy'yi eski bir sürümle kur
RUN pip install numpy==1.26.4

# requirements.txt dosyasını kopyala ve bağımlılıkları kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .
CMD ["python", "main.py"]
