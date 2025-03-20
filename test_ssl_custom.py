import requests
import ssl
from urllib3.util.ssl_ import create_urllib3_context

context = create_urllib3_context(ssl_minimum_version=ssl.TLSVersion.TLSv1_2)
context.set_ciphers("TLS_AES_256_GCM_SHA384:TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384")

url = "https://api.binance.com/api/v3/exchangeInfo"
try:
    response = requests.get(url, timeout=10, verify=True) #, context=context)
    print(f"Status Code: {response.status_code}")
    print(response.json()[:100])
except Exception as e:
    print(f"Hata: {e}")
