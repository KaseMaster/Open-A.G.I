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
│   │   └── quantum_coherence_ai.py
│   ├── api/               # API endpoints
│   │   ├── main.py
│   │   └── routes/
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── core/
│   ├── api/
│   └── integration/
├── docs/                  # Documentation
├── cli/                   # Command-line tools
├── config/                # Configuration files
├── logs/                  # Log files
├── database/              # Database files
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── README.md             # Project overview
```

## Core Modules

### Harmonic Validation (`src/core/harmonic_validation.py`)

This module implements the Recursive Φ-Resonance Validation (RΦV) algorithm:

#### Key Functions:
- `compute_spectrum(times, values)` - Computes frequency spectrum using FFT
- `compute_coherence_score(local_snapshot, remote_snapshots)` - Calculates coherence between nodes
- `recursive_validate(snapshot_bundle, threshold)` - Performs recursive validation
- `make_snapshot(node_id, times, values, secret_key)` - Creates signed snapshots
- `calculate_token_rewards(coherence_score, validator_chr_score)` - Calculates token rewards

#### Data Classes:
- `HarmonicSnapshot` - Represents a node's harmonic data
- `HarmonicProofBundle` - Bundle of snapshots with aggregated coherence

### Token Rules (`src/core/token_rules.py`)

This module defines the rules for the multi-token economy:

#### Key Functions:
- `validate_harmonic_tx(tx, config)` - Validates transactions based on coherence and CHR
- `apply_token_effects(state, tx)` - Updates ledger state after validation
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

```python
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
from src.core.harmonic_validation import make_snapshot

class TestHarmonicValidation(unittest.TestCase):
    def test_make_snapshot(self):
        """Test snapshot creation"""
        snapshot = make_snapshot(
            node_id="test-validator",
            times=[0.0, 0.1, 0.2],
            values=[1.0, 1.1, 1.2],
            secret_key="test-secret"
        )
        
        self.assertEqual(snapshot.node_id, "test-validator")
```

#### API Tests
```python
import unittest
from unittest.mock import patch
from src.api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
    
    def test_example_endpoint(self):
        """Test example endpoint"""
        response = self.app.get('/example')
        self.assertEqual(response.status_code, 200)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/core/test_harmonic_validation.py

# Run with coverage
python -m pytest --cov=src tests/
```

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

def example_function(data: List[float]) -> Dict[str, float]:
    """Example function with type hints"""
    return {"result": sum(data)}
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

1. Fork the repository
2. Create a feature branch
3. Make changes and write tests
4. Run all tests and ensure they pass
5. Update documentation if needed
6. Submit a pull request

### Code Review Guidelines

All pull requests must:
- Pass all automated tests
- Include appropriate test coverage
- Follow coding standards
- Include documentation updates
- Be reviewed by at least one maintainer

### Issue Reporting

When reporting issues, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information
- Relevant logs or error messages

## Development Environment

### Setting Up

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Run initial tests

```bash
git clone https://github.com/KaseMaster/Open-A.G.I.git
cd Open-A.G.I/quantum-currency
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests/
```

### Development Tools

Recommended development tools:
- VS Code with Python extension
- PyCharm for advanced features
- Git for version control
- Docker for containerization

### Debugging

Use the Python debugger for troubleshooting:

```python
import pdb; pdb.set_trace()
```

Or use IDE debugging features.

## API Client Development

### Python Client Example

```python
import requests

class QuantumCurrencyClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def create_snapshot(self, node_id, times, values, secret_key):
        """Create a harmonic snapshot"""
        data = {
            "node_id": node_id,
            "times": times,
            "values": values,
            "secret_key": secret_key
        }
        response = requests.post(f"{self.base_url}/api/v1/snapshot", json=data)
        return response.json()
```

## Extending the System

### Adding New Token Types

1. Update `get_token_properties()` in `token_rules.py`
2. Add token-specific logic in `apply_token_effects()`
3. Update tests to include new token type
4. Document new token properties

### Adding New Validation Rules

1. Modify `validate_harmonic_tx()` in `token_rules.py`
2. Update configuration parameters if needed
3. Add new test cases
4. Update documentation

### Integrating New AI Models

1. Create new model in `src/models/`
2. Integrate with existing AI modules
3. Add API endpoints for new functionality
4. Create tests and documentation

## Performance Considerations

### Optimization Tips

1. Use vectorized operations with NumPy
2. Cache expensive computations
3. Minimize database queries
4. Use connection pooling
5. Implement proper indexing

### Profiling

Use Python's built-in profiler:

```bash
python -m cProfile -o profile.out src/api/main.py
```

## Security Development

### Secure Coding Practices

1. Validate all input data
2. Use parameterized queries
3. Implement proper authentication
4. Sanitize output data
5. Follow principle of least privilege

### Cryptographic Security

1. Use established cryptographic libraries
2. Never implement custom crypto
3. Rotate keys regularly
4. Use secure random number generation
5. Implement proper key management

## Documentation Updates

When making changes, update:
- Code docstrings
- API documentation
- User guides
- README files
- Inline comments for complex logic

---

*This developer guide provides comprehensive information for working with the Quantum Currency system. For specific implementation details, refer to the source code and existing documentation.*