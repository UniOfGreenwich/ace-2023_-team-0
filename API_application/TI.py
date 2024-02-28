#Extra Test code, ignore

import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=SMA&symbol=BTC&interval=weekly&time_period=10&series_type=open&apikey=EO0074M34SS40R92'
r = requests.get(url)
data = r.json()

print(data)
