FROM python:3.8-slim

WORKDIR /app

# Gerekli sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    cmake \
    libc6-dev \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    cython \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip'i güncelle
RUN pip install --no-cache-dir --upgrade pip

# python-dotenv'u kur
RUN pip install --no-cache-dir python-dotenv

# Kök dizinden andirin.env dosyasını kopyala ve GITHUB_TOKEN ile depoyu klonla
COPY andirin.env .
RUN export $(grep -v '^#' andirin.env | xargs) \
    && git clone https://${GITHUB_TOKEN}@github.com/Algdn46/algdn46.git \
    && mv algdn46/talib . \
    && mv algdn46/bot.py . \
    && mv algdn46/requirements.txt . \
    && mv algdn46/andirin.env . \
    && rm -rf algdn46

# ta-lib C kütüphanesini kaynaktan kur
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib \
    && ldconfig

# talib klasöründeki Python bağlayıcılarını derle ve kur
RUN cd talib \
    && python setup.py build \
    && python setup.py install \
    && cd ..

# requirements.txt'den ta-lib satırını kaldır ve kalan bağımlılıkları kur
RUN sed '/ta-lib/d' requirements.txt > temp.txt && mv temp.txt requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Uygulamayı çalıştır
CMD ["python", "bot.py"]
