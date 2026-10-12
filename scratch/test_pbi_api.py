import urllib.request

resource_key = '6d9736f9-58e6-4d00-aecd-80e3b55ce5ad'
url = f'https://app.powerbi.com/public/routing/cluster/{resource_key}'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode('utf-8')
        print("Response text:", repr(text))
except Exception as e:
    print("Error:", e)
