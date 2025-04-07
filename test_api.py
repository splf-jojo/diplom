import requests

url = 'http://localhost:8000/predict'
files = {'file': open('demo_video.mp4', 'rb')}
response = requests.post(url, files=files)
print(response.json())
