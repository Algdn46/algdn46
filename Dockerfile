# Python 3.11 temel imajını kullan
FROM python:3.11

# Gerekli bağımlılıkları yükle
# TA-Lib derlemesi için libpthread ve diğer geliştirme araçlarını ekledim
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    libatlas-base-dev \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C kütüphanesini elle indirip kur
WORKDIR /tmp
RUN curl -L -o ta-lib-0.4.0-src.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    # Paralel inşa yerine tek thread ile inşa et (sorunları önlemek için)
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# TA-Lib Python paketini yükle
# Not: TA-Lib 0.6.3 sürümü C kütüphanesine bağlı, bu yüzden önce C kütüphanesi kurulmalı
RUN pip install --no-cache-dir TA-Lib==0.6.3

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Model dosyaları için dizin oluştur
RUN mkdir -p /data/models

# Uygulamayı başlat
CMD ["python", "bot.py"]
