# Dockerfile for Quantum Currency System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
COPY requirements-test.txt* ./

# Install dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ -f requirements-test.txt ]; then pip install --no-cache-dir -r requirements-test.txt; fi

# Copy project files
COPY . .

# Create openagi directory if it doesn't exist
RUN mkdir -p openagi

# Expose port (if needed for web services)
EXPOSE 8080

# Default command
CMD ["python", "main.py", "start-node"]
