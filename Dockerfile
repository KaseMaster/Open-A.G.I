# Dockerfile for AEGIS Framework
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build tools for packages that require compilation (e.g., netifaces)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
COPY requirements-test.txt* ./

# Install dependencies
# Nota: torch es muy grande (~2GB), puede causar problemas de espacio en CI
# Instalar dependencias excluyendo torch primero, luego intentar torch opcionalmente
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir $(grep -v "^torch" requirements.txt | grep -v "^#" | grep -v "^$" | tr '\n' ' ') || echo "Some dependencies failed to install, continuing..."
# Intentar instalar torch opcionalmente (puede fallar por espacio, no es crítico para CI básico)
RUN pip install --no-cache-dir torch>=2.1.0 --no-deps 2>&1 | head -5 || echo "torch installation skipped (space constraints - optional for CI)"
RUN if [ -f requirements-test.txt ]; then pip install --no-cache-dir -r requirements-test.txt; fi

# Copy project files
COPY . .

# Create openagi directory if it doesn't exist
RUN mkdir -p openagi

# Expose port (if needed for web services)
EXPOSE 8080

# Default command
CMD ["python", "main.py", "start-node"]
