# 🪙 Quantum Currency Deployment Guide

## Overview

This guide provides instructions for deploying the Quantum Currency system in various environments including development, testing, and production.

## Prerequisites

### System Requirements
- Python 3.9 or higher
- pip package manager
- Docker (for containerized deployment)
- Git (for source code management)
- At least 4GB RAM
- 10GB free disk space

### Required Python Packages
```bash
numpy>=1.21.0
scipy>=1.7.0
flask>=2.0.0
pytest>=6.2.0
```

## Installation Methods

### 🐍 Manual Installation

1. **Clone the repository**:
```bash
git clone https://github.com/KaseMaster/Open-A.G.I.git
cd Open-A.G.I/quantum-currency
```

2. **Create a virtual environment**:
```bash
python -m venv quantum-currency-env
source quantum-currency-env/bin/activate  # On Windows: quantum-currency-env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Run the system**:
```bash
python src/api/main.py
```

### 🐳 Docker Installation (Recommended)

1. **Build the Docker image**:
```bash
docker build -t quantum-currency .
```

2. **Run the container**:
```bash
docker run -p 5000:5000 quantum-currency
```

3. **Or use Docker Compose**:
```bash
docker-compose up -d
```

### ☸️ Kubernetes Deployment

1. **Apply the Kubernetes manifests**:
```bash
kubectl apply -f kubernetes/
```

2. **Check the deployment status**:
```bash
kubectl get deployments
kubectl get services
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Flask Configuration
FLASK_ENV=production
FLASK_APP=src/api/main.py
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///quantum_currency.db

# Security Configuration
JWT_SECRET_KEY=your-jwt-secret-key-here
HMAC_SECRET_KEY=your-hmac-secret-key-here

# Network Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=False

# Quantum Currency Specific
MINT_THRESHOLD=0.75
MIN_CHR=0.6
```

### Configuration Files

The system uses several configuration files located in the `config/` directory:

- `config/app.py` - Application configuration
- `config/database.py` - Database configuration
- `config/security.py` - Security settings
- `config/tokens.py` - Token economy parameters

## Database Setup

The Quantum Currency system uses SQLite for development and PostgreSQL for production.

### Initialize Database
```bash
python src/database/init_db.py
```

### Database Migration
```bash
python src/database/migrate.py
```

## Testing

### Run Unit Tests
```bash
python -m pytest tests/core/ -v
```

### Run API Tests
```bash
python -m pytest tests/api/ -v
```

### Run Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### Run All Tests with Coverage
```bash
python -m pytest --cov=src --cov-report=html tests/
```

## Monitoring and Observability

### Health Checks
```bash
curl http://localhost:5000/health
```

### Metrics Endpoint
```bash
curl http://localhost:5000/metrics
```

### Logging
Logs are written to:
- `logs/app.log` - Application logs
- `logs/security.log` - Security events
- `logs/transactions.log` - Transaction logs

## Security Considerations

### Key Management
- Use Hardware Security Modules (HSM) for production
- Rotate keys regularly
- Never commit secrets to version control

### Network Security
- Use HTTPS in production
- Implement firewall rules
- Restrict access to sensitive endpoints

### Data Protection
- Encrypt sensitive data at rest
- Use homomorphic encryption for privacy
- Implement proper access controls

## Scaling

### Horizontal Scaling
- Deploy multiple validator nodes
- Use load balancers for API endpoints
- Implement database replication

### Vertical Scaling
- Increase CPU and memory resources
- Optimize database queries
- Use caching mechanisms

## Backup and Recovery

### Database Backup
```bash
sqlite3 quantum_currency.db .dump > backup.sql
```

### Configuration Backup
```bash
cp .env .env.backup
cp config/*.py config/backup/
```

### Recovery Procedure
1. Restore database from backup
2. Restore configuration files
3. Restart services
4. Verify system health

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH=src
   ```

2. **Database connection issues**:
   ```bash
   # Check database file permissions
   ls -la quantum_currency.db
   # Initialize database if missing
   python src/database/init_db.py
   ```

3. **Port already in use**:
   ```bash
   # Change port in .env file
   PORT=5001
   ```

### Debugging

Enable debug mode by setting:
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

Check logs in the `logs/` directory for detailed error information.

## Performance Optimization

### Caching
- Implement Redis for frequently accessed data
- Use in-memory caching for snapshots
- Cache coherence scores for recent calculations

### Database Optimization
- Add indexes to frequently queried columns
- Use connection pooling
- Optimize query performance

### API Optimization
- Implement request batching
- Use compression for large responses
- Cache API responses where appropriate

## Maintenance

### Regular Tasks
- Monitor system health daily
- Rotate logs weekly
- Update dependencies monthly
- Review security settings quarterly

### Update Procedure
1. Backup current system
2. Pull latest changes from repository
3. Update dependencies
4. Run database migrations
5. Restart services
6. Verify functionality

## Production Checklist

Before deploying to production, ensure:

- [ ] All tests pass
- [ ] Security audit completed
- [ ] Performance benchmarks met
- [ ] Backup procedures tested
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team trained on operations

## Support

For issues and support, please:

1. Check the documentation
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Contact the development team

---

*This deployment guide provides comprehensive instructions for running the Quantum Currency system. Always test changes in a staging environment before deploying to production.*