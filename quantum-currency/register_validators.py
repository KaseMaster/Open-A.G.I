import requests
import json

# Register multiple validators
validators = [
    {
        "validator_id": "validator-002",
        "staked_atr": 1500.0,
        "initial_coherence": 0.85
    },
    {
        "validator_id": "validator-003",
        "staked_atr": 1200.0,
        "initial_coherence": 0.9
    },
    {
        "validator_id": "validator-004",
        "staked_atr": 800.0,
        "initial_coherence": 0.75
    },
    {
        "validator_id": "validator-005",
        "staked_atr": 2000.0,
        "initial_coherence": 0.88
    }
]

url = "http://localhost:5000/uhes/governance/validator/register"
headers = {
    "Content-Type": "application/json"
}

for validator in validators:
    try:
        response = requests.post(url, data=json.dumps(validator), headers=headers)
        print(f"Registering {validator['validator_id']}: Status {response.status_code}")
        print(f"Response: {response.text}")
        print("---")
    except Exception as e:
        print(f"Error registering {validator['validator_id']}: {e}")