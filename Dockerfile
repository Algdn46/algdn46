# Base image

FROM python:3.7.3-slim-stretch

RUN apt-get update

# Installing docker

RUN apt-get update
RUN apt-get update && apt-get install -y \
    gcc \
    && curl -L https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz | tar xvz

WORKDIR /ta-lib
# numpy needs to be installed before TA-Lib
RUN pip3 install 'numpy==1.16.2' \
  && pip3 install 'TA-Lib==0.4.17'

RUN cd .. && rm -rf ta-lib/
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive
# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set entrypoint
CMD ["python", "bot.py"]
