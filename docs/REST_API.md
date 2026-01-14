# Quantum Currency REST API

## Overview
This REST API provides endpoints for the quantum currency system, including snapshot generation, coherence calculation, minting, and ledger queries.

## Endpoints

### POST /snapshot
Generate a signed snapshot

Request:
```json
{
  "node_id": "string",
  "times": [number],
  "values": [number],
  "secret_key": "string"
}
```

Response:
```json
{
  "node_id": "string",
  "timestamp": number,
  "spectrum_hash": "string",
  "signature": "string"
}
```

### POST /coherence
Calculate coherence score between local and remote snapshots

Request:
```json
{
  "local": {
    "node_id": "string",
    "times": [number],
    "values": [number],
    "secret_key": "string"
  },
  "remotes": [
    {
      "node_id": "string",
      "times": [number],
      "values": [number],
      "secret_key": "string"
    }
  ]
}
```

Response:
```json
{
  "coherence_score": number
}
```

### POST /mint
Validate and mint FLX tokens

Request:
```json
{
  "id": "string",
  "type": "harmonic",
  "action": "mint",
  "token": "FLX",
  "sender": "string",
  "receiver": "string",
  "amount": number,
  "aggregated_cs": number,
  "sender_chr": number
}
```

Response (success):
```json
{
  "status": "accepted",
  "ledger": {
    "balances": {},
    "chr": {}
  }
}
```

Response (failure):
```json
{
  "status": "rejected",
  "reason": "string"
}
```

### GET /ledger
Get current ledger state

Response:
```json
{
  "balances": {},
  "chr": {}
}
```

## Running the API

```bash
python -m openagi.rest_api
```

The API will start on http://localhost:5000