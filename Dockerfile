FROM python:3.7

# Gerekli paketleri yükle (Debian tabanlı sistemler için)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    cmake \
    libta-lib0 \
    libta-lib-dev \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ana uygulamayı kopyala
COPY . .

# Çalıştırma komutu
CMD ["python", "bot.py"]

