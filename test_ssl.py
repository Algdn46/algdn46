import requests

url = "https://api.binance.com/api/v3/exchangeInfo"
try:
    response = requests.get(url, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(response.json()[:100])
except Exception as e:
    print(f"Hata: {e}")
