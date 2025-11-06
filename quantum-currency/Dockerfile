# Quantum Currency Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "src/api/main.py"]