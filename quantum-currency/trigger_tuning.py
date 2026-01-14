import requests

# Trigger AI-driven coherence tuning
url = "http://localhost:5000/uhes/tuning/trigger"

try:
    response = requests.post(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")