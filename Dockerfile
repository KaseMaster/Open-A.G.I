# Dockerfile for Quantum Currency System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-test.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy project files
COPY . .

# Create openagi directory if it doesn't exist
RUN mkdir -p openagi

# Expose port (if needed for web services)
EXPOSE 8080

# Default command
CMD ["python", "scripts/demo_emulation.py"]