import urllib.request
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/tatsuya-tkst/nikkei225/master/nikkei225.csv"

try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        body = response.read().decode('utf-8')
        
    codes = []
    for line in body.split('\n')[1:]: # skip header
        if line.strip():
            code = line.split(',')[0].strip()
            if code: codes.append(code + ".T")
            
    # Add TOPIX Core30 + Large70 to reach ~300
    additional_codes = [
        '4661.T', '7974.T', '6594.T', '6861.T', '6954.T', '6981.T', '8035.T',
        '4307.T', '2127.T', '2413.T', '3382.T', '3391.T', '4528.T', '4568.T',
        '4578.T', '4612.T', '4689.T', '4704.T', '4768.T', '4901.T', '4911.T',
        '5108.T', '5401.T', '5411.T', '5713.T', '5802.T', '6098.T', '6146.T',
        '6273.T', '6301.T', '6326.T', '6367.T', '6501.T', '6503.T', '6506.T',
        '6701.T', '6702.T', '6723.T', '6752.T', '6758.T', '6869.T', '6902.T',
        '6920.T', '7011.T', '7201.T', '7203.T', '7267.T', '7269.T', '7309.T',
        '7733.T', '7741.T', '7751.T', '8001.T', '8002.T', '8031.T', '8053.T',
        '8058.T', '8306.T', '8308.T', '8309.T', '8316.T', '8411.T', '8591.T',
        '8604.T', '8630.T', '8725.T', '8750.T', '8766.T', '8801.T', '8802.T',
        '8830.T', '9020.T', '9021.T', '9022.T', '9101.T', '9104.T', '9202.T',
        '9432.T', '9433.T', '9434.T', '9735.T', '9843.T', '9983.T', '9984.T'
    ]
    
    final_tickers = sorted(list(set(codes + additional_codes)))
    with open('tickers.txt', 'w') as f:
        f.write(',\n'.join([f'    \"{t}\"' for t in final_tickers[:300]]))
    print(f"Generated {len(final_tickers[:300])} tickers")
except Exception as e:
    print(f"Error fetching data: {e}")
