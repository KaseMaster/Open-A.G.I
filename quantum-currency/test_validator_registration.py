import requests
import json

# Test validator registration
url = "http://localhost:5000/uhes/governance/validator/register"
data = {
    "validator_id": "validator-001",
    "staked_atr": 1000.0,
    "initial_coherence": 0.8
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")