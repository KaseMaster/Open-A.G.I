# Contributing to Quantum Currency Implementation

Thank you for your interest in contributing to the Quantum Currency Implementation! We welcome contributions from the community to help improve and expand this innovative quantum-harmonic currency system.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Security Vulnerabilities](#security-vulnerabilities)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](.github/CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Open-A.G.I.git
   cd Open-A.G.I
   ```
3. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Process

1. Ensure you have the latest changes from the main branch:
   ```bash
   git fetch origin
   git checkout feature/quantum-currency-beta
   git merge origin/feature/quantum-currency-beta
   ```

2. Create a feature branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes, following the coding standards below

4. Write tests for your changes

5. Run the test suite to ensure nothing is broken:
   ```bash
   python -m pytest tests/
   ```

6. Commit your changes with a clear, descriptive commit message

7. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a pull request against the `feature/quantum-currency-beta` branch

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Limit lines to 88 characters (compatible with Black formatter)
- Use descriptive variable and function names
- Write docstrings for all public classes and functions

### Type Hints

- Use type hints for function parameters and return values
- Use `typing` module for complex types

### Code Organization

- Group imports at the top of the file
- Separate standard library, third-party, and local imports with blank lines
- Use `__all__` to define public API in modules

### Example

```python
import hashlib
import time
from typing import List, Dict, Optional

def compute_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of data.
    
    Args:
        data: Bytes to hash
        
    Returns:
        Hexadecimal representation of the hash
    """
    return hashlib.sha256(data).hexdigest()
```

## Testing

### Test Structure

- Place unit tests in `tests/core/`
- Place API tests in `tests/api/`
- Place integration tests in `tests/integration/`

### Writing Tests

- Use `unittest` or `pytest` framework
- Write clear, descriptive test method names
- Test both positive and negative cases
- Use mocks and fixtures where appropriate
- Aim for high code coverage (>95%)

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/core/test_harmonic_validation.py

# Run with coverage
python -m pytest --cov=openagi --cov-report=html
```

## Documentation

### Docstrings

- Use Google-style docstrings
- Document all parameters, return values, and exceptions
- Include examples for complex functions

### README Updates

- Update README.md when adding new features
- Include usage examples for new functionality

### API Documentation

- Update REST API documentation when modifying endpoints
- Maintain consistency in API documentation format

## Submitting Changes

### Pull Request Process

1. Ensure your code follows the coding standards
2. Write clear, concise commit messages
3. Include tests for new functionality
4. Update documentation as needed
5. Describe your changes in the pull request description
6. Link to any related issues

### Pull Request Template

When submitting a pull request, please include:

- Description of changes
- Related issues (if any)
- Testing performed
- Screenshots (if applicable)

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, etc.)
- Any relevant logs or error messages

### Feature Requests

When requesting a feature, please include:

- Clear, descriptive title
- Detailed description of the feature
- Use cases for the feature
- Potential implementation approaches (if known)

## Security Vulnerabilities

If you discover a security vulnerability, please send an email to [security@openagi.org](mailto:security@openagi.org) instead of using the issue tracker. We take security seriously and will respond as quickly as possible.

Please do not publicly disclose the vulnerability until we have had a chance to address it.

## Community

Join our community discussions on [Discord](#) or [Telegram](#) to connect with other contributors and users.

Thank you for contributing to the Quantum Currency Implementation!