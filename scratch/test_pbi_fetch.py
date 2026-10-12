import urllib.request
import re
import json
import base64

url = 'https://app.powerbi.com/view?r=eyJrIjoiNmQ5NzM2ZjktNThlNi00ZDAwLWFlY2QtODBlM2I1NWNlNWFkIiwidCI6ImQ1NmVjMTliLTE3ZGEtNGQ5Zi05YjdhLTY4Y2VhOWNkYTAyNiIsImMiOjl9'

# Decode r parameter (JWT / Base64)
# eyJrIjoiNmQ5NzM2ZjktNThlNi00ZDAwLWFlY2QtODBlM2I1NWNlNWFkIiwidCI6ImQ1NmVjMTliLTE3ZGEtNGQ5Zi05YjdhLTY4Y2VhOWNkYTAyNiIsImMiOjl9
r_param = 'eyJrIjoiNmQ5NzM2ZjktNThlNi00ZDAwLWFlY2QtODBlM2I1NWNlNWFkIiwidCI6ImQ1NmVjMTliLTE3ZGEtNGQ5Zi05YjdhLTY4Y2VhOWNkYTAyNiIsImMiOjl9'
decoded_r = base64.b64decode(r_param + '==').decode('utf-8')
print("Decoded r parameter:", decoded_r)

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        print("HTTP Status:", resp.status)
        print("HTML length:", len(html))
        # Look for cluster URL or app configuration
        for line in html.splitlines():
            if 'app.' in line or 'telemetry' in line or 'report' in line or 'cluster' in line:
                if len(line) < 300:
                    print("Line:", line.strip())
except Exception as e:
    print("Fetch error:", e)
