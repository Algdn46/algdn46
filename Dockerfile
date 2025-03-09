# Base image
FROM python:3.9.18-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-rc1.tar.gz && \
    tar -xzf ta-lib-0.4.0-rc1.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-rc1.tar.gz

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data volume
RUN mkdir /data && chmod 755 /data

# Run application
CMD ["python", "-
