# AEGIS Framework - Advanced API Documentation

## Table of Contents
1. [Security API](#security-api)
2. [Performance API](#performance-api)
3. [Monitoring API](#monitoring-api)
4. [Federated Learning API](#federated-learning-api)
5. [Blockchain API](#blockchain-api)
6. [Model Serving API](#model-serving-api)

---

## Security API

### Zero-Knowledge Proofs

#### Create Zero-Knowledge Proof
```http
POST /api/v1/security/zkp
```

**Request Body**
```json
{
  "secret": "base64_encoded_secret",
  "statement": "authenticate:user_123",
  "verifier_id": "auth_service",
  "proof_type": "range_proof",
  "parameters": {
    "min_value": 0,
    "max_value": 1000
  }
}
```

**Response**
```json
{
  "proof_id": "zkp_abc123",
  "commitment": "base64_encoded_commitment",
  "challenge": "base64_encoded_challenge",
  "timestamp": 1700000000,
  "expires_at": 1700003600
}
```

#### Verify Zero-Knowledge Proof
```http
POST /api/v1/security/zkp/verify
```

**Request Body**
```json
{
  "proof_id": "zkp_abc123",
  "public_statement": "authenticate:user_123",
  "proof_data": {
    "commitment": "base64_encoded_commitment",
    "challenge": "base64_encoded_challenge",
    "response": "base64_encoded_response"
  }
}
```

**Response**
```json
{
  "valid": true,
  "verification_time_ms": 12.5,
  "security_level": "high"
}
```

### Homomorphic Encryption

#### Encrypt Value
```http
POST /api/v1/security/homomorphic/encrypt
```

**Request Body**
```json
{
  "value": 42,
  "metadata": {
    "type": "salary",
    "employee_id": "emp_123"
  }
}
```

**Response**
```json
{
  "encrypted_value": "base64_encoded_ciphertext",
  "public_key": "base64_encoded_public_key",
  "nonce": "base64_encoded_nonce",
  "metadata": {
    "type": "salary",
    "employee_id": "emp_123"
  }
}
```

#### Decrypt Value
```http
POST /api/v1/security/homomorphic/decrypt
```

**Request Body**
```json
{
  "encrypted_value": "base64_encoded_ciphertext",
  "private_key_required": true
}
```

**Response**
```json
{
  "decrypted_value": 42,
  "decryption_time_ms": 5.2,
  "success": true
}
```

#### Homomorphic Addition
```http
POST /api/v1/security/homomorphic/add
```

**Request Body**
```json
{
  "encrypted_values": [
    "base64_encoded_ciphertext_1",
    "base64_encoded_ciphertext_2"
  ]
}
```

**Response**
```json
{
  "result": "base64_encoded_sum_ciphertext",
  "operation_time_ms": 8.7,
  "success": true
}
```

### Secure Multi-Party Computation

#### Generate Secret Shares
```http
POST /api/v1/security/smc/shares
```

**Request Body**
```json
{
  "secret": 12345,
  "threshold": 3,
  "total_shares": 5,
  "participants": ["party_1", "party_2", "party_3", "party_4", "party_5"]
}
```

**Response**
```json
{
  "shares": {
    "party_1": "base64_share_1",
    "party_2": "base64_share_2",
    "party_3": "base64_share_3",
    "party_4": "base64_share_4",
    "party_5": "base64_share_5"
  },
  "commitments": ["base64_commitment_1", "base64_commitment_2"],
  "generation_time_ms": 15.3
}
```

#### Reconstruct Secret
```http
POST /api/v1/security/smc/reconstruct
```

**Request Body**
```json
{
  "shares": {
    "party_1": "base64_share_1",
    "party_3": "base64_share_3",
    "party_5": "base64_share_5"
  },
  "threshold_met": true
}
```

**Response**
```json
{
  "reconstructed_secret": 12345,
  "verification_passed": true,
  "reconstruction_time_ms": 12.8
}
```

### Differential Privacy

#### Privatize Query
```http
POST /api/v1/security/differential-privacy/query
```

**Request Body**
```json
{
  "query_type": "count",
  "data": [1, 1, 0, 1, 0, 1, 1],
  "epsilon": 1.0,
  "delta": 1e-5,
  "sensitivity": 1.0
}
```

**Response**
```json
{
  "private_result": 4.2,
  "noise_added": 0.2,
  "privacy_budget_used": 1.0,
  "confidence_interval": [3.8, 4.6]
}
```

---

## Performance API

### Memory Optimization

#### Get Memory Stats
```http
GET /api/v1/performance/memory
```

**Response**
```json
{
  "cache_size": 8500,
  "pools": {
    "consensus_objects": 45,
    "network_buffers": 120,
    "crypto_keys": 23
  },
  "total_accesses": 125000,
  "hit_rate": 0.95,
  "memory_mb": 128.5
}
```

#### Clear Memory Cache
```http
DELETE /api/v1/performance/memory/cache
```

**Response**
```json
{
  "cleared_items": 8500,
  "freed_memory_mb": 45.2,
  "operation_time_ms": 12.3
}
```

### Concurrency Optimization

#### Get Concurrency Stats
```http
GET /api/v1/performance/concurrency
```

**Response**
```json
{
  "running_tasks": 23,
  "queue_size": 5,
  "semaphores": {
    "network_connections": 8,
    "database_queries": 15,
    "file_operations": 20
  },
  "task_metrics": {
    "network_success": 1250,
    "network_error": 3,
    "database_success": 890,
    "database_error": 12
  }
}
```

#### Create Semaphore
```http
POST /api/v1/performance/concurrency/semaphore
```

**Request Body**
```json
{
  "name": "ml_training",
  "limit": 4,
  "timeout_seconds": 30
}
```

**Response**
```json
{
  "created": true,
  "semaphore_id": "sem_ml_training_123",
  "current_limit": 4
}
```

### Network Optimization

#### Get Network Stats
```http
GET /api/v1/performance/network
```

**Response**
```json
{
  "connection_pools": {
    "consensus_nodes": 15,
    "database": 8,
    "external_apis": 12
  },
  "active_connections": {
    "consensus_nodes": 12,
    "database": 6,
    "external_apis": 8
  },
  "message_batches": {
    "consensus_broadcast": 3,
    "data_sync": 7
  },
  "compression_enabled": true
}
```

#### Batch Messages
```http
POST /api/v1/performance/network/batch
```

**Request Body**
```json
{
  "destination": "consensus_nodes",
  "messages": [
    {"type": "prepare", "data": "base64_data_1"},
    {"type": "commit", "data": "base64_data_2"},
    {"type": "prepare", "data": "base64_data_3"}
  ],
  "max_batch_size": 50,
  "timeout_seconds": 0.1
}
```

**Response**
```json
{
  "batch_ready": true,
  "batch_size": 3,
  "batch_data": "base64_batched_messages",
  "compression_ratio": 0.75
}
```

---

## Monitoring API

### Security Monitoring

#### Get Security Metrics
```http
GET /api/v1/monitoring/security
```

**Response**
```json
{
  "timestamp": 1700000000,
  "enabled_features": {
    "zero_knowledge_proofs": true,
    "homomorphic_encryption": true,
    "secure_mpc": true,
    "differential_privacy": true
  },
  "operation_metrics": {
    "zk_proofs_generated": 1250,
    "zk_proofs_verified": 1245,
    "encryption_operations": 5600,
    "decryption_operations": 5598,
    "smc_operations": 230,
    "dp_queries": 890
  },
  "success_rates": {
    "zk_proofs": 0.996,
    "encryption": 1.0,
    "decryption": 0.999,
    "smc": 0.987,
    "dp": 1.0
  },
  "parties_in_smc": 15
}
```

#### Get Security Events
```http
GET /api/v1/monitoring/security/events?limit=50&severity=warning
```

**Response**
```json
{
  "events": [
    {
      "timestamp": 1700000000,
      "event_type": "zk_proof_failed",
      "severity": "warning",
      "message": "ZK proof verification failed for node_123",
      "details": {
        "node_id": "node_123",
        "proof_id": "zkp_abc123",
        "attempt": 1
      }
    }
  ],
  "total_events": 1,
  "time_range": {
    "start": 1699996400,
    "end": 1700000000
  }
}
```

### Performance Monitoring

#### Get Performance Metrics
```http
GET /api/v1/monitoring/performance
```

**Response**
```json
{
  "uptime_seconds": 3600,
  "operation_metrics": {
    "zk_proof_generation": {
      "total_calls": 1250,
      "avg_time_ms": 12.5,
      "min_time_ms": 8.2,
      "max_time_ms": 25.7,
      "success_rate": 0.996,
      "error_rate": 0.004
    },
    "homomorphic_encryption": {
      "total_calls": 5600,
      "avg_time_ms": 5.2,
      "min_time_ms": 3.1,
      "max_time_ms": 12.8,
      "success_rate": 1.0,
      "error_rate": 0.0
    }
  },
  "memory_stats": {
    "cache_hits": 45000,
    "cache_misses": 2300,
    "hit_rate": 0.951,
    "memory_mb": 128.5
  },
  "concurrency_stats": {
    "running_tasks": 23,
    "queue_size": 5,
    "task_success_rate": 0.985
  }
}
```

---

## Federated Learning API

### Advanced Security Integration

#### Start Privacy-Preserving Training
```http
POST /api/v1/fl/training/start
```

**Request Body**
```json
{
  "model_name": "secure_mnist_classifier",
  "config": {
    "aggregation_strategy": "secure_aggregation",
    "num_rounds": 100,
    "clients_per_round": 10,
    "local_epochs": 5,
    "learning_rate": 0.01,
    "security_config": {
      "homomorphic_encryption": true,
      "differential_privacy": {
        "epsilon": 1.0,
        "delta": 1e-5
      },
      "secure_aggregation": {
        "threshold": 7,
        "total_participants": 10
      }
    }
  }
}
```

**Response**
```json
{
  "job_id": "fl_job_abc123",
  "status": "started",
  "security_features": {
    "homomorphic_encryption": true,
    "differential_privacy": true,
    "secure_aggregation": true
  },
  "estimated_completion_time": 1800
}
```

#### Get Training Security Report
```http
GET /api/v1/fl/training/{job_id}/security-report
```

**Response**
```json
{
  "job_id": "fl_job_abc123",
  "privacy_preserved": true,
  "security_features_used": {
    "homomorphic_encryption": {
      "operations": 12500,
      "avg_time_per_operation_ms": 5.2,
      "success_rate": 1.0
    },
    "differential_privacy": {
      "queries": 890,
      "epsilon_consumed": 0.8,
      "noise_added": true
    },
    "secure_aggregation": {
      "rounds_completed": 45,
      "participants_verified": 450,
      "zero_knowledge_authentications": 450
    }
  },
  "data_protection_metrics": {
    "raw_data_exposure": 0.0,
    "encrypted_transmission": 1.0,
    "privacy_budget_remaining": 0.2
  }
}
```

---

## Blockchain API

### Secure Consensus with Advanced Features

#### Propose Secure Block
```http
POST /api/v1/blockchain/blocks/propose
```

**Request Body**
```json
{
  "block_data": {
    "transactions": ["tx_1", "tx_2", "tx_3"],
    "metadata": {
      "timestamp": 1700000000,
      "creator": "node_123"
    }
  },
  "consensus_config": {
    "security_level": "paranoid",
    "zero_knowledge_authentication": true,
    "homomorphic_validation": true
  },
  "previous_hash": "abc123def456"
}
```

**Response**
```json
{
  "proposal_id": "proposal_xyz789",
  "security_features": {
    "zk_authentication": true,
    "homomorphic_validation": true,
    "encrypted_payload": true
  },
  "validation_required": ["prepare", "commit"],
  "timeout_seconds": 30
}
```

#### Validate Secure Proposal
```http
POST /api/v1/blockchain/blocks/validate
```

**Request Body**
```json
{
  "proposal_id": "proposal_xyz789",
  "validator_id": "validator_456",
  "validation_type": "prepare",
  "security_config": {
    "zero_knowledge_proof": {
      "secret": "validator_secret_456",
      "statement": "validate_proposal:proposal_xyz789"
    },
    "homomorphic_verification": true
  }
}
```

**Response**
```json
{
  "validation_id": "val_789xyz",
  "approved": true,
  "security_checks": {
    "zk_authentication": "passed",
    "homomorphic_validation": "passed",
    "timestamp_validation": "passed"
  },
  "signature": "base64_signature"
}
```

---

## Model Serving API

### Secure Model Inference

#### Perform Encrypted Inference
```http
POST /api/v1/models/{model_id}/predict/secure
```

**Request Body**
```json
{
  "encrypted_input": "base64_encrypted_input_data",
  "encryption_scheme": "homomorphic_rsa",
  "security_config": {
    "preserve_privacy": true,
    "add_differential_privacy": false
  }
}
```

**Response**
```json
{
  "encrypted_prediction": "base64_encrypted_prediction",
  "inference_time_ms": 45.2,
  "security_features": {
    "homomorphic_encryption": true,
    "input_privacy_preserved": true,
    "output_encrypted": true
  },
  "confidence_metrics": {
    "homomorphic_accuracy": 0.945,
    "privacy_guarantee": "strong"
  }
}
```

#### Get Model Security Profile
```http
GET /api/v1/models/{model_id}/security-profile
```

**Response**
```json
{
  "model_id": "model_abc123",
  "security_features": {
    "input_encryption": true,
    "output_encryption": true,
    "differential_privacy": true,
    "access_control": "rbac",
    "audit_logging": true
  },
  "privacy_metrics": {
    "differential_privacy_epsilon": 1.0,
    "differential_privacy_delta": 1e-5,
    "data_exposure_risk": "minimal"
  },
  "compliance": {
    "gdpr_compliant": true,
    "hipaa_compliant": false,
    "soc2_compliant": true
  },
  "last_security_audit": 1699999999,
  "next_audit_scheduled": 1700603599
}
```

---

## WebSocket API

### Real-time Security Events
```javascript
// Connect to security events WebSocket
const ws = new WebSocket('ws://localhost:8080/ws/security-events');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  switch(data.event_type) {
    case 'security_alert':
      console.log('Security Alert:', data.message);
      console.log('Severity:', data.severity);
      break;
      
    case 'zk_proof_generated':
      console.log('ZK Proof Generated:', data.proof_id);
      break;
      
    case 'homomorphic_operation':
      console.log('Homomorphic Operation:', data.operation_type);
      console.log('Time:', data.elapsed_time_ms, 'ms');
      break;
  }
};
```

### Real-time Performance Metrics
```javascript
// Connect to performance metrics WebSocket
const perfWs = new WebSocket('ws://localhost:8080/ws/performance-metrics');

perfWs.onmessage = function(event) {
  const data = JSON.parse(event.data);
  
  if (data.metrics_type === 'security_operations') {
    console.log('Security Operations Per Second:', data.ops_per_second);
    console.log('Average Latency:', data.avg_latency_ms, 'ms');
  }
};
```

---

## Error Responses

### Common Security Errors

```http
403 Forbidden
```
```json
{
  "error": "insufficient_permissions",
  "message": "Access denied: insufficient permissions for security operation",
  "required_permission": "security.admin",
  "current_user_role": "user"
}
```

```http
422 Unprocessable Entity
```
```json
{
  "error": "security_validation_failed",
  "message": "Security validation failed: invalid zero-knowledge proof",
  "validation_errors": [
    "Proof commitment mismatch",
    "Challenge response verification failed"
  ]
}
```

```http
429 Too Many Requests
```
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded for security operations",
  "limit": 100,
  "window_seconds": 60,
  "retry_after_seconds": 45
}
```

---

## Authentication

### API Key Authentication
```http
Authorization: Bearer YOUR_API_KEY_HERE
```

### Zero-Knowledge Authentication
```http
Authorization: ZKP proof_id="zkp_abc123", challenge_response="base64_response"
```

### Certificate-Based Authentication
```http
Authorization: Certificate cert="base64_certificate", signature="base64_signature"
```

---

## Rate Limiting

### Security Operations Limits
- **Zero-Knowledge Proofs**: 1000/hour
- **Homomorphic Encryption**: 5000/hour
- **Secure MPC Operations**: 100/hour
- **Differential Privacy Queries**: 1000/hour

### Performance Operations Limits
- **Memory Operations**: 10000/hour
- **Concurrency Operations**: 5000/hour
- **Network Operations**: 10000/hour

---

## Best Practices

### Security Implementation
1. **Always use HTTPS** for security-sensitive operations
2. **Rotate API keys** regularly
3. **Implement proper access controls** using RBAC
4. **Enable audit logging** for all security operations
5. **Monitor security metrics** in real-time

### Performance Optimization
1. **Use connection pooling** for database and network operations
2. **Implement proper caching** strategies
3. **Monitor memory usage** and implement garbage collection
4. **Use batch operations** when possible
5. **Implement proper error handling** and retry logic

### Monitoring and Alerting
1. **Set up alerts** for security incidents
2. **Monitor performance metrics** continuously
3. **Implement health checks** for all services
4. **Use dashboards** for real-time visibility
5. **Set up automated reporting** for compliance
