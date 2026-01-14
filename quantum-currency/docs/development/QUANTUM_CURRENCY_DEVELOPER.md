# 🪙 Quantum Currency Developer Guide

## Overview

This guide provides information for developers who want to contribute to the Quantum Currency system or build applications on top of it.

## Project Structure

```
quantum-currency/
├── src/                    # Source code
│   ├── core/              # Core modules
│   │   ├── harmonic_validation.py
│   │   ├── token_rules.py
│   │   └── validator_staking.py
│   ├── models/            # Data models
│   │   ├── harmonic_wallet.py
│   │   ├── coherence_attunement_layer.py
│   │   └── quantum_coherence_ai.py
│   ├── api/               # API endpoints
│   │   ├── main.py
│   │   └── routes/
│   ├── simulation/        # Multi-node simulation framework
│   │   └── multi_node_simulator.py
│   ├── dashboard/         # Dashboard and UX components
│   │   └── dashboard_app.py
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── core/
│   ├── api/
│   ├── integration/
│   ├── simulation/
│   └── dashboard/
├── docs/                  # Documentation
├── cli/                   # Command-line tools
│   └── quantum_cli.py
├── config/                # Configuration files
├── logs/                  # Log files
├── database/              # Database files
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── README.md             # Project overview
```

## Core Modules

### Coherence Attunement Layer (`src/models/coherence_attunement_layer.py`)

This new module (v0.2.0) implements the Ω-State recursion and modulator functions:

#### Key Functions:
- `compute_omega_state(token_data, sentiment_data, semantic_data, attention_data)` - Computes multi-dimensional Ω-state vectors
- `compute_recursive_coherence(omega_states)` - Calculates recursive coherence with penalty components
- `validate_dimensional_consistency(omega_state)` - Ensures dimensional integrity
- `compute_modulator(coherence_score)` - Adaptive weighting function

#### Data Classes:
- `OmegaState` - Represents a node's multi-dimensional coherence state
- `CoherencePenalties` - Encapsulates cosine, entropy, and variance penalties

### Harmonic Validation (`src/core/harmonic_validation.py`)

This module implements the Recursive Φ-Resonance Validation (RΦV) algorithm with CAL integration:

#### Key Functions:
- `compute_spectrum(times, values)` - Computes frequency spectrum using FFT
- `compute_coherence_score(local_snapshot, remote_snapshots)` - Calculates coherence between nodes
- `recursive_validate(snapshot_bundle, threshold)` - Performs recursive validation with Ω-state integration
- `make_snapshot(node_id, times, values, secret_key)` - Creates signed snapshots
- `calculate_token_rewards(coherence_score, validator_chr_score)` - Calculates token rewards

#### Data Classes:
- `HarmonicSnapshot` - Represents a node's harmonic data with Ω-state
- `HarmonicProofBundle` - Bundle of snapshots with aggregated coherence

### Token Rules (`src/core/token_rules.py`)

This module defines the rules for the multi-token economy with harmonic gating:

#### Key Functions:
- `validate_harmonic_tx(tx, config)` - Validates transactions based on coherence and CHR
- `apply_token_effects(state, tx)` - Updates ledger state with harmonic gating
- `get_token_properties(token_type)` - Returns properties of a token type

#### Token Properties:
1. **FLX**: Transferable utility token
2. **CHR**: Non-transferable reputation token
3. **PSY**: Semi-transferable synchronization token
4. **ATR**: Stakable stability token
5. **RES**: Multiplicative expansion token

### Validator Staking (`src/core/validator_staking.py`)

This module implements the validator staking system:

#### Key Functions:
- `create_staking_position()` - Creates staking positions
- `unstake_tokens()` - Unstakes tokens
- `get_staking_apr()` - Gets staking APR
- `get_validator_info()` - Gets validator information

## API Development

### Adding New Endpoints

1. Create a new route file in `src/api/routes/`
2. Define the endpoint function with proper documentation
3. Add the route to `src/api/main.py`
4. Create corresponding tests in `tests/api/`

### Example Endpoint Implementation

```
# src/api/routes/example.py
from flask import Blueprint, request, jsonify

example_bp = Blueprint('example', __name__)

@example_bp.route('/example', methods=['GET'])
def example_endpoint():
    """Example endpoint documentation"""
    return jsonify({"message": "Hello, Quantum Currency!"})
```

## Testing

### Test Structure

The test suite is organized as follows:
- `tests/core/` - Unit tests for core modules
- `tests/api/` - API endpoint tests
- `tests/integration/` - Integration tests

### Writing Tests

#### Core Module Tests
```python
import unittest
from src.models.coherence_attunement_layer import CoherenceAttunementLayer
from src.core.harmonic_validation import make_snapshot

class TestCoherenceAttunementLayer(unittest.TestCase):
    def setUp(self):
        self.cal = CoherenceAttunementLayer()
    
    def test_compute_omega_state(self):
        """Test Ω-state computation"""
        omega = self.cal.compute_omega_state(
            token_data={"rate": 5.0},
            sentiment_data={"energy": 0.7},
            semantic_data={"shift": 0.3},
            attention_data=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        self.assertIsNotNone(omega)
        self.assertEqual(omega.token_rate, 5.0)

class TestHarmonicValidation(unittest.TestCase):
    def test_make_snapshot(self):
        """Test snapshot creation with Ω-state"""
        snapshot = make_snapshot(
            node_id="test-validator",
            times=[0.0, 0.1, 0.2],
            values=[1.0, 1.1, 1.2],
            secret_key="test-secret"
        )
        
        self.assertEqual(snapshot.node_id, "test-validator")
        self.assertIsNotNone(snapshot.omega_state)
```

#### API Tests
```python
import unittest
from unittest.mock import patch
from src.api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
    
    def test_omega_endpoint(self):
        """Test Ω-state endpoint"""
        response = self.app.post('/ai/omega', json={
            "token_data": {"rate": 5.0},
            "sentiment_data": {"energy": 0.7}
        })
        self.assertEqual(response.status_code, 200)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/core/test_coherence_attunement_layer.py

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test class
python -m pytest tests/core/test_coherence_attunement_layer.py::TestCoherenceAttunementLayer

# Run integration tests
python -m pytest tests/integration/
```

### Testing Protocols (v0.2.0)

#### Dimensional Consistency Test
Validates that all Ω-state components remain within safe bounds:
- All Ω components must be in [-10, +10]
- Modulator arguments must be in [-10, +10]
- Modulator outputs must be in [exp(-10), exp(10)]

#### Harmonic Shock Recovery Test
Validates system resilience:
- Ψ must recover ≥ 0.70 within 50 steps after a Ψ drop > 0.3
- Recovery must follow stable, predictable patterns

#### AI Coherence Regression Test
Validates AI alignment:
- AI actions must result in positive ΔΨ 75% of the time when Ψ < 0.8
- AI must default to "grounding" actions when Ψ is low

## Code Quality

### Style Guide

Follow PEP 8 style guidelines:
- Use 4 spaces for indentation
- Limit lines to 79 characters
- Use descriptive variable names
- Write docstrings for all public functions

### Type Hints

Use type hints for all function parameters and return values:

```python
from typing import List, Dict, Tuple
from src.models.coherence_attunement_layer import OmegaState

def example_function(data: List[float]) -> Dict[str, float]:
    """Example function with type hints"""
    return {"result": sum(data)}

def compute_omega_states(inputs: List[Dict]) -> List[OmegaState]:
    """Compute multiple Ω-states with type safety"""
    cal = CoherenceAttunementLayer()
    return [cal.compute_omega_state(**input_data) for input_data in inputs]
```

### Linting

Run code quality checks before committing:

```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy --package src
```

## Contributing

### Pull Request Process