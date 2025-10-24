# AEGIS Framework - Complete API Reference

## Table of Contents
1. [Model Serving API](#model-serving-api)
2. [Federated Learning API](#federated-learning-api)
3. [Blockchain API](#blockchain-api)
4. [Monitoring API](#monitoring-api)
5. [Security Middleware](#security-middleware)

---

## Model Serving API

### Base URL
```
http://localhost:8000
```

### Authentication
All endpoints require API key authentication via header:
```
Authorization: Bearer <api_key>
```

---

### GET /health
Health check endpoint

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-24T18:00:00",
  "models_registered": 5,
  "active_training_jobs": 2
}
```

---

### GET /api/v1/models
List all registered models

**Response**
```json
[
  {
    "model_id": "abc123def456",
    "model_name": "mnist_classifier",
    "version": "1.0.0",
    "created_at": "2025-10-24T18:00:00",
    "accuracy": 0.95,
    "num_parameters": 1234567,
    "framework": "pytorch",
    "tags": {
      "dataset": "mnist",
      "architecture": "cnn"
    }
  }
]
```

---

### GET /api/v1/models/{model_id}
Get metadata for a specific model

**Parameters**
- `model_id` (path): Model identifier

**Response**
```json
{
  "model_id": "abc123def456",
  "model_name": "mnist_classifier",
  "version": "1.0.0",
  "created_at": "2025-10-24T18:00:00",
  "accuracy": 0.95,
  "num_parameters": 1234567,
  "framework": "pytorch",
  "tags": {}
}
```

---

### POST /api/v1/inference
Perform model inference

**Request Body**
```json
{
  "model_id": "abc123def456",
  "input_data": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  ],
  "batch_size": 32
}
```

**Response**
```json
{
  "model_id": "abc123def456",
  "predictions": [1, 0, 1],
  "confidence_scores": [0.95, 0.87, 0.92],
  "inference_time_ms": 23.5,
  "timestamp": "2025-10-24T18:00:00"
}
```

**Example (Python)**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/inference",
    json={
        "model_id": "abc123def456",
        "input_data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Inference time: {result['inference_time_ms']}ms")
```

---

### DELETE /api/v1/models/{model_id}
Delete a model from registry

**Parameters**
- `model_id` (path): Model identifier

**Response**
```json
{
  "message": "Model abc123def456 deleted successfully"
}
```

---

## Federated Learning API

### POST /api/v1/training/start
Start a new federated learning training job

**Request Body**
```json
{
  "model_name": "federated_mnist",
  "aggregation_strategy": "scaffold",
  "num_rounds": 10,
  "clients_per_round": 5,
  "local_epochs": 3,
  "learning_rate": 0.01,
  "config_override": {
    "mu": 0.01,
    "differential_privacy": true
  }
}
```

**Parameters**
- `model_name` (string): Name for the trained model
- `aggregation_strategy` (string): One of `fedavg`, `fedprox`, `scaffold`, `fedopt`, `fednova`
- `num_rounds` (integer): Total training rounds (1-1000)
- `clients_per_round` (integer): Clients participating per round (2-100)
- `local_epochs` (integer): Local training epochs (1-10)
- `learning_rate` (float): Learning rate (0.0001-1.0)
- `config_override` (object, optional): Override configuration parameters

**Response**
```json
{
  "job_id": "job_abc123",
  "message": "Training job started successfully",
  "status_endpoint": "/api/v1/training/status/job_abc123"
}
```

**Example (Python)**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/training/start",
    json={
        "model_name": "my_fl_model",
        "aggregation_strategy": "scaffold",
        "num_rounds": 10,
        "clients_per_round": 5,
        "local_epochs": 3,
        "learning_rate": 0.01
    }
)

job_id = response.json()["job_id"]
print(f"Training job started: {job_id}")
```

---

### GET /api/v1/training/status/{job_id}
Get status of a training job

**Parameters**
- `job_id` (path): Training job identifier

**Response**
```json
{
  "job_id": "job_abc123",
  "status": "running",
  "current_round": 5,
  "total_rounds": 10,
  "current_loss": 0.234,
  "current_accuracy": 0.912,
  "started_at": "2025-10-24T18:00:00",
  "updated_at": "2025-10-24T18:05:00"
}
```

**Example (Python)**
```python
import requests
import time

job_id = "job_abc123"

while True:
    response = requests.get(
        f"http://localhost:8000/api/v1/training/status/{job_id}"
    )
    status = response.json()
    
    print(f"Round {status['current_round']}/{status['total_rounds']}")
    print(f"Loss: {status['current_loss']:.4f}, Accuracy: {status['current_accuracy']:.4f}")
    
    if status['status'] != 'running':
        break
    
    time.sleep(5)
```

---

### POST /api/v1/training/stop/{job_id}
Stop a training job

**Parameters**
- `job_id` (path): Training job identifier

**Response**
```json
{
  "message": "Training job job_abc123 stopped successfully"
}
```

---

### GET /api/v1/training/metrics/{job_id}
Get detailed training metrics

**Parameters**
- `job_id` (path): Training job identifier

**Response**
```json
{
  "job_id": "job_abc123",
  "current_round": 10,
  "metrics": {
    "loss": [0.8, 0.6, 0.4, 0.3, 0.25, 0.23, 0.22, 0.21, 0.20, 0.19],
    "accuracy": [0.70, 0.75, 0.82, 0.86, 0.89, 0.90, 0.91, 0.92, 0.92, 0.93],
    "training_time": [12.3, 11.8, 11.5, 11.2, 11.0, 10.9, 10.8, 10.7, 10.6, 10.5]
  },
  "client_history": [
    {
      "round": 1,
      "num_clients": 5,
      "total_samples": 5000,
      "loss": 0.8,
      "accuracy": 0.70,
      "training_time": 12.3
    }
  ]
}
```

---

## Aggregation Strategies

### FedAvg (Federated Averaging)
Simple weighted average of client model updates.

**Use case**: Standard federated learning, homogeneous data

**Example**
```python
{
  "aggregation_strategy": "fedavg"
}
```

---

### FedProx (Federated Proximal)
Adds proximal term to handle heterogeneous data.

**Use case**: Non-IID data distribution, system heterogeneity

**Configuration**
```python
{
  "aggregation_strategy": "fedprox",
  "config_override": {
    "mu": 0.01  # Proximal term coefficient
  }
}
```

---

### SCAFFOLD
Uses control variates for variance reduction.

**Use case**: Reduced communication rounds, heterogeneous clients

**Example**
```python
{
  "aggregation_strategy": "scaffold"
}
```

---

### FedOpt (Federated Optimization)
Server-side adaptive optimization.

**Use case**: Improved convergence, adaptive learning

**Configuration**
```python
{
  "aggregation_strategy": "fedopt",
  "config_override": {
    "server_learning_rate": 0.5,
    "server_momentum": 0.9
  }
}
```

---

### FedNova
Normalized averaging for heterogeneous local steps.

**Use case**: Variable client computation, heterogeneous updates

**Example**
```python
{
  "aggregation_strategy": "fednova"
}
```

---

## Security Middleware

### Rate Limiting

Rate limits protect against abuse and DDoS attacks.

**Default Limits**
- 100 requests per minute
- 20 burst allowance
- 5-minute block after 3 violations

**Configuration**
```python
from aegis.security.middleware import SecurityMiddleware, RateLimitRule

middleware = SecurityMiddleware()

# Custom rule for specific endpoint
middleware.rate_limiter.set_rule(
    "/api/v1/inference",
    RateLimitRule(
        max_requests=50,
        window_seconds=60,
        burst_allowance=10
    )
)

# Check rate limit
allowed, message = middleware.check_request_security(
    client_id="192.168.1.100",
    endpoint="/api/v1/inference",
    params={"model_id": "abc123"}
)

if not allowed:
    print(f"Request blocked: {message}")
```

---

### Input Validation

Validates and sanitizes all user inputs.

**Example**
```python
from aegis.security.middleware import InputValidator

validator = InputValidator()

# Validate string
is_valid, error = validator.validate_string(
    value="user_input_123",
    field_type="alphanumeric",
    min_length=3,
    max_length=50
)

if not is_valid:
    print(f"Validation error: {error}")

# Validate integer
is_valid, error = validator.validate_integer(
    value=42,
    min_value=1,
    max_value=100
)

# Validate email
is_valid, error = validator.validate_string(
    value="user@example.com",
    field_type="email"
)

# Sanitize string
safe_string = validator.sanitize_string(
    "<script>alert('xss')</script>Hello"
)
print(safe_string)  # "Hello"
```

---

## Complete Examples

### Example 1: Train and Deploy FL Model

```python
import requests
import time

API_BASE = "http://localhost:8000"

# 1. Start federated training
response = requests.post(
    f"{API_BASE}/api/v1/training/start",
    json={
        "model_name": "production_model_v1",
        "aggregation_strategy": "scaffold",
        "num_rounds": 20,
        "clients_per_round": 10,
        "local_epochs": 5,
        "learning_rate": 0.01
    }
)

job_id = response.json()["job_id"]
print(f"Training job started: {job_id}")

# 2. Monitor training progress
while True:
    response = requests.get(
        f"{API_BASE}/api/v1/training/status/{job_id}"
    )
    status = response.json()
    
    print(f"Round {status['current_round']}/{status['total_rounds']} - "
          f"Loss: {status['current_loss']:.4f}, "
          f"Accuracy: {status['current_accuracy']:.4f}")
    
    if status['status'] == 'completed':
        print("Training completed!")
        break
    
    time.sleep(10)

# 3. Get final metrics
response = requests.get(
    f"{API_BASE}/api/v1/training/metrics/{job_id}"
)
metrics = response.json()

print(f"\nFinal Metrics:")
print(f"  Loss: {metrics['metrics']['loss'][-1]:.4f}")
print(f"  Accuracy: {metrics['metrics']['accuracy'][-1]:.4f}")

# 4. List models
response = requests.get(f"{API_BASE}/api/v1/models")
models = response.json()

print(f"\n{len(models)} models registered")
for model in models:
    print(f"  - {model['model_name']} (v{model['version']})")
```

---

### Example 2: Inference with Security

```python
import requests
from aegis.security.middleware import SecurityMiddleware

API_BASE = "http://localhost:8000"

# Setup security middleware
middleware = SecurityMiddleware()

# Client information
client_id = "192.168.1.100"
endpoint = "/api/v1/inference"

# Prepare request
params = {
    "model_id": "abc123def456",
    "input_data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
}

# Security check
allowed, message = middleware.check_request_security(
    client_id=client_id,
    endpoint=endpoint,
    params=params
)

if not allowed:
    print(f"Request blocked: {message}")
    exit(1)

# Perform inference
response = requests.post(
    f"{API_BASE}{endpoint}",
    json=params
)

result = response.json()

print(f"Predictions: {result['predictions']}")
print(f"Confidence: {result['confidence_scores']}")
print(f"Inference time: {result['inference_time_ms']}ms")
```

---

## Error Handling

All API endpoints return standard error responses:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

**Common Status Codes**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource doesn't exist)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

**Example Error Handling**
```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/api/v1/inference",
        json={"model_id": "invalid"},
        timeout=10
    )
    response.raise_for_status()
    
    result = response.json()
    print(result)
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Message: {e.response.json()['detail']}")
    
except requests.exceptions.Timeout:
    print("Request timed out")
    
except Exception as e:
    print(f"Error: {e}")
```

---

## Rate Limit Headers

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1698163200
```

---

## Changelog

### Version 1.0.0 (2025-10-24)
- Initial API release
- Model serving endpoints
- Federated learning with 5 algorithms
- Security middleware with rate limiting
- Input validation and sanitization
