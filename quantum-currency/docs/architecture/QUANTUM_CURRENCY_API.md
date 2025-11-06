# 🪙 Quantum Currency API Reference

## Overview

The Quantum Currency API provides RESTful endpoints for interacting with the quantum-harmonic currency system. It enables snapshot generation, coherence calculation, token minting, and ledger queries.

With the v0.2.0 upgrade, the API includes new endpoints for the Coherence Attunement Layer (CAL) and Ω-State computations.

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

**Description**: Generate a signed harmonic snapshot from time series data with Ω-State integration.

**Request Body**:
```json
{
  "node_id": "string",
  "times": [0.0, 0.1, 0.2, ...],
  "values": [1.0, 1.1, 1.2, ...],
  "secret_key": "string",
  "token_data": {
    "rate": 5.0
  },
  "sentiment_data": {
    "energy": 0.7
  },
  "semantic_data": {
    "shift": 0.3
  },
  "attention_data": [0.1, 0.2, 0.3, 0.4, 0.5]
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
  "omega_state": {
    "timestamp": 1234567890.123,
    "token_rate": 5.0,
    "sentiment_energy": 0.7,
    "semantic_shift": 0.3,
    "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
    "coherence_score": 0.87,
    "modulator": 1.23,
    "time_delay": 0.67
  },
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

**Description**: Calculate coherence score between local and remote snapshots with CAL enhancement.

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
    "omega_state": {
      "timestamp": 1234567890.123,
      "token_rate": 5.0,
      "sentiment_energy": 0.7,
      "semantic_shift": 0.3,
      "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
      "coherence_score": 0.87,
      "modulator": 1.23,
      "time_delay": 0.67
    },
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
      "omega_state": {
        "timestamp": 1234567890.123,
        "token_rate": 4.8,
        "sentiment_energy": 0.65,
        "semantic_shift": 0.28,
        "meta_attention_spectrum": [0.12, 0.22, 0.32, 0.42, 0.52],
        "coherence_score": 0.85,
        "modulator": 1.18,
        "time_delay": 0.65
      },
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
  "coherence_score": 0.95,
  "penalties": {
    "cosine_penalty": 0.05,
    "entropy_penalty": 0.03,
    "variance_penalty": 0.02
  },
  "validation_status": "ACCEPTED"
}
```

### Mint Tokens
```
POST /mint
```

**Description**: Validate and mint FLX tokens based on coherence score and CHR reputation with CAL enhancement.

**Request Body**:
```json
{
  "sender": "string",
  "receiver": "string",
  "amount": 100.0,
  "token": "FLX",
  "action": "mint",
  "aggregated_cs": 0.95,
  "sender_chr": 0.85,
  "omega_state": {
    "timestamp": 1234567890.123,
    "token_rate": 5.0,
    "sentiment_energy": 0.7,
    "semantic_shift": 0.3,
    "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
    "coherence_score": 0.87,
    "modulator": 1.23,
    "time_delay": 0.67
  }
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
  "reason": "coherence or CHR too low",
  "required_coherence": 0.7,
  "current_coherence": 0.65
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
    "receiver": "validator-2",
    "amount": 100.0,
    "token": "FLX",
    "timestamp": 1234567890.123,
    "status": "completed"
  }
]
```

## CAL-Enhanced AI Endpoints

### Compute Ω-State
```
POST /ai/omega
```

**Description**: Compute multi-dimensional Ω-state vector for cognitive coherence assessment.

**Request Body**:
```json
{
  "token_data": {
    "rate": 5.0
  },
  "sentiment_data": {
    "energy": 0.7
  },
  "semantic_data": {
    "shift": 0.3
  },
  "attention_data": [0.1, 0.2, 0.3, 0.4, 0.5]
}
```

**Response**:
```json
{
  "omega_state": {
    "timestamp": 1234567890.123,
    "token_rate": 5.0,
    "sentiment_energy": 0.7,
    "semantic_shift": 0.3,
    "meta_attention_spectrum": [0.1, 0.2, 0.3, 0.4, 0.5],
    "coherence_score": 0.87,
    "modulator": 1.23,
    "time_delay": 0.67
  },
  "dimensional_consistency": "VALID"
}
```

### Get Coherence Health
```
GET /ai/health
```

**Description**: Get health status of Quantum Coherence AI with Ω-state metrics.

**Response**:
```json
{
  "status": "healthy",
  "coherence_score": 0.92,
  "nodes_monitored": 5,
  "average_omega_stability": 0.88,
  "last_updated": 1234567890.123
}
```

### Predict Coherence Trends
```
POST /ai/predict
```

**Description**: Get AI-driven coherence predictions based on historical Ω-states.

**Request Body**:
```json
{
  "prediction_window": 100,
  "factors": ["token_rate", "sentiment_energy"]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "timestamp": 1234567990.123,
      "predicted_coherence": 0.91,
      "confidence": 0.85
    }
  ],
  "trend": "STABLE"
}
```

### Run Autonomous Validation
```
POST /ai/autonomous
```

**Description**: Run autonomous validator orchestration cycles with CAL enhancement.

**Request Body**:
```json
{
  "cycle_count": 10,
  "validation_threshold": 0.8
}
```

**Response**:
```json
{
  "cycles_completed": 10,
  "successful_validations": 9,
  "failed_validations": 1,
  "average_coherence": 0.89
}
```

## Security Endpoints

### Generate Secure Keys
```
POST /security/keys
```

**Description**: Generate quantum-resistant key pairs for secure transactions.

**Request Body**:
```json
{
  "key_type": "ed25519",
  "purpose": "validation"
}
```

**Response**:
```json
{
  "public_key": "string",
  "key_id": "string",
  "expiration": 1234567890.123
}
```

## Error Responses

All endpoints may return the following error responses:

```json
{
  "error": "Invalid request format",
  "code": 400
}
```

```json
{
  "error": "Unauthorized access",
  "code": 401
}
```

```json
{
  "error": "Resource not found",
  "code": 404
}
```

```json
{
  "error": "Internal server error",
  "code": 500
}
```

---