import requests

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
data = {"features": [600, 40, 3, 60000, 2, 1, 0, 0, 1, 0, 1]}

response = requests.post(url, json=data, headers=headers)
print(response.json())  # Print API response
