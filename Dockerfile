# Base image
FROM python:3.9.18-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget https://downloads.sourceforge.net/project/ta-lib/ta-lib-0.3.0-src.tar.gz && \
    tar -xzf ta-lib-0.3.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.3.0-src.tar.gz ta-lib

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data volume with proper permissions
RUN mkdir -p /data && chmod 755 /data

# Set entrypoint
CMD ["python", "bot.py"]
