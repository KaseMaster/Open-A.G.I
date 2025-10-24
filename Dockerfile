# Multi-stage Dockerfile for AEGIS Framework
# Optimized for production with minimal size (<500MB target)
# Programador Principal: Jose Gómez alias KaseMaster
# Versión: 2.0.0

# ============================================================================
# Stage 1: Builder - Install dependencies and compile
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies without cache to reduce size
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="kasemaster@protonmail.com"
LABEL version="2.0.0"
LABEL description="AEGIS Framework - Distributed AI with Blockchain"

# Create non-root user for security
RUN groupadd -r aegis && useradd -r -g aegis aegis

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=aegis:aegis src/ /app/src/
COPY --chown=aegis:aegis main.py /app/
COPY --chown=aegis:aegis config/ /app/config/
COPY --chown=aegis:aegis examples/ /app/examples/

# Copy only necessary root files
COPY --chown=aegis:aegis requirements.txt /app/
COPY --chown=aegis:aegis pyproject.toml /app/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AEGIS_ENV=production \
    AEGIS_LOG_LEVEL=INFO \
    AEGIS_DASHBOARD_PORT=8080

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/certs && \
    chown -R aegis:aegis /app

# Switch to non-root user
USER aegis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["python", "main.py", "start-node", "--dry-run"]

# ============================================================================
# Build instructions:
# docker build -t aegis-framework:2.0.0 .
# docker run -p 8080:8080 aegis-framework:2.0.0
# ============================================================================
