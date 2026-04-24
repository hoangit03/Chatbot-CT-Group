import base64
import requests
import time

file_path = 'shared_data/data_input/01d86b98-0cf8-4c33-b7cf-e5a03fdd1114_page0.png'
url = 'http://localhost:8088/layout-parsing'

print(f'Reading file: {file_path}')
with open(file_path, 'rb') as f:
    file_b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'file_b64': file_b64,
    'file_name': 'test_image.png'
}

print(f'Sending request to {url} ...')
start = time.time()
try:
    response = requests.post(url, json=payload, timeout=60)
    elapsed = time.time() - start
    print(f'Status Code: {response.status_code}')
    print(f'Time elapsed: {elapsed:.2f} seconds')
    
    if response.status_code == 200:
        data = response.json()
        print('Success!')
        print(f'Number of markdown blocks extracted: {len(data.get("markdown", []))}')
        print('Sample content from first block:')
        if data.get("markdown"):
            print(data["markdown"][0][:200] + '...')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Request failed: {e}')
