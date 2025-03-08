FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for TA-Lib compilation)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Compile and install TA-Lib C library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Update pip
RUN pip install --upgrade pip

# Install TA-Lib Python package (after C library is installed)
RUN pip install TA-Lib==0.4.28

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "bot.py"]
