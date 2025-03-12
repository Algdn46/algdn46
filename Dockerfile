FROM python:3.11-slim
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libta-lib0 libta-lib0-dev libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install ta-lib
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "bot.py"]

