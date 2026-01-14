# 🪙 Quantum Currency API Reference

## Overview

The Quantum Currency API provides RESTful endpoints for interacting with the quantum-harmonic currency system. It enables snapshot generation, coherence calculation, token minting, and ledger queries.

## Base URL

```
http://localhost:5000/api/v1
```

## Authentication

API endpoints do not require authentication for local development. In production environments, JWT tokens are used for authentication and authorization.

## Core Endpoints

### Generate Snapshot
```
POST /snapshot
```

**Description**: Generate a signed harmonic snapshot from time series data.

**Request Body**:
```json
{
  "node_id": "string",
  "times": [0.0, 0.1, 0.2, ...],
  "values": [1.0, 1.1, 1.2, ...],
  "secret_key": "string"
}
```

**Response**:
```json
{
  "node_id": "string",
  "timestamp": 1234567890.123,
  "times": [0.0, 0.1, 0.2, ...],
  "values": [1.0, 1.1, 1.2, ...],
  "spectrum": [[0.0, 1.0], [0.1, 0.9], ...],
  "spectrum_hash": "string",
  "CS": 0.95,
  "phi_params": {
    "phi": 1.618033988749895,
    "lambda": 0.618033988749895,
    "tau": 1.0
  },
  "signature": "string"
}
```

### Calculate Coherence
```
POST /coherence
```

**Description**: Calculate coherence score between local and remote snapshots.

**Request Body**:
```json
{
  "local": {
    "node_id": "string",
    "timestamp": 1234567890.123,
    "times": [0.0, 0.1, 0.2, ...],
    "values": [1.0, 1.1, 1.2, ...],
    "spectrum": [[0.0, 1.0], [0.1, 0.9], ...],
    "spectrum_hash": "string",
    "CS": 0.0,
    "phi_params": {
      "phi": 1.618033988749895,
      "lambda": 0.618033988749895,
      "tau": 1.0
    },
    "signature": "string"
  },
  "remotes": [
    {
      "node_id": "string",
      "timestamp": 1234567890.123,
      "times": [0.0, 0.1, 0.2, ...],
      "values": [1.0, 1.1, 1.2, ...],
      "spectrum": [[0.0, 1.0], [0.1, 0.9], ...],
      "spectrum_hash": "string",
      "CS": 0.0,
      "phi_params": {
        "phi": 1.618033988749895,
        "lambda": 0.618033988749895,
        "tau": 1.0
      },
      "signature": "string"
    }
  ]
}
```

**Response**:
```json
{
  "coherence_score": 0.95
}
```

### Mint Tokens
```
POST /mint
```

**Description**: Validate and mint FLX tokens based on coherence score and CHR reputation.

**Request Body**:
```json
{
  "sender": "string",
  "receiver": "string",
  "amount": 100.0,
  "token": "FLX",
  "action": "mint",
  "aggregated_cs": 0.95,
  "sender_chr": 0.85
}
```

**Response** (Success):
```json
{
  "status": "accepted",
  "ledger": {
    "balances": {
      "validator-1": {
        "FLX": 1100.0,
        "CHR": 500.0,
        "PSY": 200.0,
        "ATR": 300.0,
        "RES": 50.0
      }
    },
    "chr": {
      "validator-1": 0.85
    }
  }
}
```

**Response** (Failure):
```json
{
  "status": "rejected",
  "reason": "coherence or CHR too low"
}
```

### Get Ledger State
```
GET /ledger
```

**Description**: Retrieve the current ledger state including all balances and CHR scores.

**Response**:
```json
{
  "balances": {
    "validator-1": {
      "FLX": 1000.0,
      "CHR": 500.0,
      "PSY": 200.0,
      "ATR": 300.0,
      "RES": 50.0
    },
    "validator-2": {
      "FLX": 1500.0,
      "CHR": 750.0,
      "PSY": 150.0,
      "ATR": 400.0,
      "RES": 75.0
    }
  },
  "chr": {
    "validator-1": 0.85,
    "validator-2": 0.92
  }
}
```

### Get Transaction History
```
GET /transactions
```

**Description**: Retrieve all transaction history from the database.

**Response**:
```json
[
  {
    "sender": "validator-1",
    "receiver": "validator-1",
    "amount": 100.0,
    "token": "FLX",
    "action": "mint",
    "aggregated_cs": 0.95,
    "sender_chr": 0.85,
    "timestamp": 1234567890.123
  }
]
```

### Get Snapshot History
```
GET /snapshots
```

**Description**: Retrieve all snapshot history from the database.

**Response**:
```json
[
  {
    "node_id": "string",
    "timestamp": 1234567890.123,
    "times": [0.0, 0.1, 0.2, ...],
    "values": [1.0, 1.1, 1.2, ...],
    "spectrum": [[0.0, 1.0], [0.1, 0.9], ...],
    "spectrum_hash": "string",
    "CS": 0.95,
    "phi_params": {
      "phi": 1.618033988749895,
      "lambda": 0.618033988749895,
      "tau": 1.0
    },
    "signature": "string"
  }
]
```

## AI Integration Endpoints

### Get AI Health Status
```
GET /ai/health
```

**Description**: Get the health status of the Quantum Coherence AI system.

**Response**:
```json
{
  "status": "healthy",
  "uptime": 3600,
  "last_prediction": 1234567890.123,
  "accuracy": 0.98
}
```

### Get AI Predictions
```
POST /ai/predict
```

**Description**: Get AI-driven coherence predictions.

**Request Body**:
```json
{
  "node_id": "string",
  "historical_data": [
    {
      "timestamp": 1234567890.123,
      "coherence_score": 0.95
    }
  ]
}
```

**Response**:
```json
{
  "predicted_coherence": 0.92,
  "confidence": 0.85,
  "recommendations": [
    "Increase sampling frequency",
    "Optimize node positioning"
  ]
}
```

### Run Autonomous Orchestration
```
POST /ai/autonomous
```

**Description**: Run an autonomous validator orchestration cycle.

**Response**:
```json
{
  "cycle_id": "string",
  "status": "completed",
  "nodes_optimized": 5,
  "improvements": {
    "avg_coherence": 0.03,
    "network_stability": 0.05
  }
}
```

## Token Properties

### FLX (Φlux)
- **Transferable**: Yes
- **Stakable**: No
- **Convertible**: Yes
- **Utility**: Network operations, energy representation

### CHR (Coheron)
- **Transferable**: No
- **Stakable**: No
- **Convertible**: Yes (to ATR and RES)
- **Utility**: Ethical alignment, reputation, governance

### PSY (ΨSync)
- **Transferable**: Semi (with fees)
- **Stakable**: No
- **Convertible**: Yes (to ATR)
- **Utility**: Node synchronization, behavioral incentives

### ATR (Attractor)
- **Transferable**: Yes
- **Stakable**: Yes
- **Convertible**: Yes (to RES)
- **Utility**: Network stability, anchoring transitions

### RES (Resonance)
- **Transferable**: Yes
- **Stakable**: No
- **Convertible**: No
- **Utility**: Network expansion, multiplicative rewards

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid request data
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error responses follow this format:
```json
{
  "error": "Descriptive error message",
  "code": "ERROR_CODE"
}
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:
- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

## Versioning

The API uses semantic versioning. Breaking changes will result in a new major version number.

## Examples

### Python Client Example
```python
import requests
import json

# Generate a snapshot
snapshot_data = {
    "node_id": "validator-1",
    "times": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "values": [1.0, 1.1, 1.2, 1.1, 1.0, 0.9],
    "secret_key": "my-secret-key"
}

response = requests.post("http://localhost:5000/api/v1/snapshot", json=snapshot_data)
snapshot = response.json()

# Calculate coherence with another node
coherence_data = {
    "local": snapshot,
    "remotes": [other_snapshot]
}

response = requests.post("http://localhost:5000/api/v1/coherence", json=coherence_data)
coherence_score = response.json()["coherence_score"]

# Mint tokens if coherence is high enough
if coherence_score > 0.75:
    mint_data = {
        "sender": "validator-1",
        "receiver": "validator-1",
        "amount": 100.0,
        "token": "FLX",
        "action": "mint",
        "aggregated_cs": coherence_score,
        "sender_chr": 0.85
    }
    
    response = requests.post("http://localhost:5000/api/v1/mint", json=mint_data)
    if response.json()["status"] == "accepted":
        print("Tokens minted successfully!")
```

---

*For more information about the Quantum Currency system, please refer to the architecture documentation and source code.*