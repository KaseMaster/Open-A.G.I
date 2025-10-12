# AEGIS Framework - Production Dockerfile
# Advanced Encrypted Governance and Intelligence System
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add metadata labels
LABEL maintainer="AEGIS Development Team <dev@aegis-project.org>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="AEGIS Framework" \
      org.label-schema.description="Advanced Encrypted Governance and Intelligence System" \
      org.label-schema.url="https://github.com/AEGIS-Project/AEGIS" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/AEGIS-Project/AEGIS" \
      org.label-schema.vendor="AEGIS Project" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for building
RUN groupadd -r aegis && useradd -r -g aegis aegis

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip cache purge

# Copy source code
COPY . .

# Change ownership to aegis user
RUN chown -R aegis:aegis /app

# Switch to non-root user
USER aegis

# Build the application
RUN python setup.py build

# Production stage
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    AEGIS_ENV=production \
    AEGIS_CONFIG_PATH=/app/config/config.yml \
    AEGIS_LOG_LEVEL=INFO \
    AEGIS_DATA_DIR=/app/data \
    AEGIS_BACKUP_DIR=/app/backups \
    AEGIS_CERT_DIR=/app/certs \
    AEGIS_DASHBOARD_PORT=8080

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    openssl \
    tor \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for production
RUN groupadd -r aegis && useradd -r -g aegis -d /app -s /bin/bash aegis

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/backups /app/certs /app/config \
    && chown -R aegis:aegis /app

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=aegis:aegis /app .

# Install only production dependencies
RUN pip install --no-cache-dir -e . && \
    pip cache purge

# Copy configuration files
COPY --chown=aegis:aegis config/config.example.yml config/config.yml

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Initialize directories\n\
mkdir -p /app/data /app/logs /app/backups /app/certs\n\
\n\
# Set proper permissions\n\
chown -R aegis:aegis /app/data /app/logs /app/backups /app/certs\n\
\n\
# Execute command\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Create healthcheck script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8080/health || exit 1' > /healthcheck.sh && chmod +x /healthcheck.sh

# Switch to non-root user
USER aegis

# Expose ports
EXPOSE 8080 8181 9090 9091

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /healthcheck.sh

# Set up volumes for persistent data
VOLUME ["/app/data", "/app/logs", "/app/backups", "/app/certs"]

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]

# Default command
CMD ["python", "main.py"]