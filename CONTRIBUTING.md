# Contributing to AEGIS Framework

First off, thank you for considering contributing to AEGIS Framework! It's people like you that make AEGIS such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)

---

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Git**
- **Docker 24+** (optional)
- **GitHub CLI** (optional, for advanced workflows)

### Quick Setup

```bash
# 1. Clone and navigate
git clone https://github.com/KaseMaster/Open-A.G.I.git
cd Open-A.G.I

# 2. Run quick start script
bash scripts/quickstart.sh

# 3. Or manual setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

---

## Development Setup

### Install Development Dependencies

```bash
pip install black isort flake8 mypy pytest-cov bandit pip-audit
```

### Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

---

## How to Contribute

### Workflow

1. **Create a branch** from `main`
2. **Implement changes** and add tests
3. **Run local checks**:
   ```bash
   # Linting
   flake8 src/ tests/ --max-line-length=100
   
   # Tests
   pytest tests/ -v
   
   # Security
   bandit -r src/
   pip-audit
   
   # Docker (optional)
   docker build -t aegis-test .
   ```
4. **Open a Pull Request** using the template

---

## Coding Standards

### Python Style

- **Line length**: 100 characters
- **Formatting**: Use `black`
- **Import sorting**: Use `isort`
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all public APIs

### Example

```python
from typing import List, Optional

def process_data(items: List[bytes], validate: bool = True) -> bool:
    """
    Process a list of items.
    
    Args:
        items: List of data items to process
        validate: Whether to validate items
        
    Returns:
        True if all items processed successfully
    """
    return all(item for item in items)
```

### Running Formatters

```bash
# Format
black src/ tests/

# Sort imports
isort src/ tests/

# Check
flake8 src/ tests/ --max-line-length=100
```

---

## Testing Guidelines

### Test Requirements

- **Coverage**: Aim for >85%
- **Structure**: Use AAA pattern (Arrange, Act, Assert)
- **Naming**: Functions start with `test_`
- **Isolation**: Tests must be independent

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/aegis --cov-report=html

# Specific test
pytest tests/test_e2e_basic_flow.py -v
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests passing
- [ ] Code formatted and linted
- [ ] Documentation updated
- [ ] Tests added for new features
- [ ] No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
Describe tests performed

## Checklist
- [ ] Tests passing
- [ ] Linters passing
- [ ] Docs updated
```

---

## Commit Messages

Follow conventional commits:

```
type(scope): subject

Examples:
- feat(crypto): add quantum-resistant encryption
- fix(consensus): resolve PBFT timeout issue
- docs(api): update CryptoEngine examples
- test(e2e): add integration test for Merkle tree
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## CI/CD

Our CI/CD pipeline runs:

- **Lint & Test**: Windows and Ubuntu, Python 3.11-3.13
- **Security**: Bandit, pip-audit, Safety, Gitleaks
- **Docker**: Multi-arch build (amd64/arm64)
- **Benchmarks**: Performance regression checks
- **Release**: Automated publishing to GHCR

---

## Reporting Bugs

### Template

```markdown
**Describe the bug**
Clear description

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Environment**
- OS: Ubuntu 22.04
- Python: 3.11.2
- AEGIS: 2.0.0

**Expected vs Actual**
What should happen vs what happened
```

---

## Security

- **Never commit secrets**
- **Report vulnerabilities privately**: kasemaster@protonmail.com
- **Keep dependencies updated**

---

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/KaseMaster/Open-A.G.I/issues)
- **Email**: kasemaster@protonmail.com

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to AEGIS Framework! 🚀**
