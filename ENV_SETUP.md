# Environment Setup Guide

## Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<tu-user>/Open-A.G.I.git
cd Open-A.G.I
git checkout -b feature/harmonic-currency
```

### 2. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n quantum-currency python=3.9
conda activate quantum-currency
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r requirements-test.txt

# Install development tools
pip install black flake8 mypy pytest-cov
```

### 4. Verify Installation

```bash
# Run a simple test to verify the environment
python -c "import numpy, scipy; print('Scientific libraries OK')"

# Run unit tests
pytest tests/ -v

# Run harmonic validation tests
pytest tests/test_harmonic_validation.py -v
```

### 5. Run Demo

```bash
# Run the 3-node validation demo
python scripts/demo_emulation.py

# Or using the shell script
bash scripts/demo_emulation.sh

# Run the harmonic-validated mint demo
python scripts/demo_mint_flex.py

# Run the REST API
python -m openagi.rest_api

# Test the REST API
python scripts/test_rest_api.py
```

## Docker Setup (Alternative)

If you prefer to use Docker:

```bash
# Build the container
docker build -t quantum-currency .

# Run the container
docker run -it quantum-currency

# Run tests in container
docker run -it quantum-currency pytest tests/ -v
```

## Development Tools

### Code Formatting

```bash
# Format code with black
black .

# Check code style with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=openagi --cov-report=html

# Run specific test file
pytest tests/test_harmonic_validation.py
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure all requirements are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Import errors**: Ensure you're in the project root directory

3. **Permission issues**: On Linux/Mac, you might need to make scripts executable
   ```bash
   chmod +x scripts/*.sh
   ```

### Environment Variables

The following environment variables can be set:

- `PYTHONPATH`: Add project root to Python path
- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

Example:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=INFO
```

## Next Steps

1. Run the demo script to verify everything works
2. Explore the harmonic validation module
3. Run tests to ensure code quality
4. Begin development on your feature branch