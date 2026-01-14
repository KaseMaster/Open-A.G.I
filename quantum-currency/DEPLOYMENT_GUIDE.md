# Quantum Currency Production Deployment Guide

This guide provides comprehensive instructions for deploying Quantum Currency in a production environment with high availability, security hardening, and auto-healing capabilities.

## 🏗️ Architecture Overview

The production deployment consists of:

1. **Load Balancer** (HAProxy/ALB) - Traffic entry point, SSL termination
2. **Nginx** - Reverse Proxy (per node)
3. **Gunicorn/Flask** - WSGI application runner (λ(t) Coherence Engine)
4. **Systemd** - Process Supervisor
5. **PostgreSQL** - State storage (Managed High-Availability Cluster)
6. **Prometheus** - Monitoring and alerting
7. **Grafana** - Visualization dashboard
8. **Alertmanager** - Notification system

## 🚀 Deployment Options

### Option 1: Native Linux Deployment

#### Prerequisites
- Ubuntu 20.04+ or CentOS 8+
- Python 3.9+
- Docker (optional but recommended)
- Nginx
- Systemd

#### Steps

1. **Create dedicated user and group:**
   ```bash
   sudo useradd --system --no-create-home --shell /bin/false quantum_user
   ```

2. **Set up project directory:**
   ```bash
   sudo mkdir -p /opt/quantum-currency
   sudo chown quantum_user:quantum_user /opt/quantum-currency
   sudo chmod 755 /opt/quantum-currency
   ```

3. **Copy project files:**
   ```bash
   sudo cp -r . /opt/quantum-currency/
   sudo chown -R quantum_user:quantum_user /opt/quantum-currency
   sudo chmod -R 755 /opt/quantum-currency
   ```

4. **Install dependencies:**
   ```bash
   cd /opt/quantum-currency
   sudo -u quantum_user pip3 install -r requirements.txt
   ```

5. **Configure Systemd service:**
   ```bash
   sudo cp systemd/quantum-gunicorn.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable quantum-gunicorn
   sudo systemctl start quantum-gunicorn
   ```

6. **Configure Nginx:**
   ```bash
   sudo cp nginx/quantum-currency.conf /etc/nginx/sites-available/
   sudo ln -sf /etc/nginx/sites-available/quantum-currency.conf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

7. **Set up auto-healing:**
   ```bash
   sudo cp systemd/quantum-healing.service /etc/systemd/system/
   sudo cp systemd/quantum-healing.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable quantum-healing.timer
   sudo systemctl start quantum-healing.timer
   ```

### Option 2: Docker Deployment

#### Prerequisites
- Docker Engine 20.10+
- Docker Compose 1.29+

#### Steps

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

2. **Scale services as needed:**
   ```bash
   docker-compose -f docker-compose.production.yml up -d --scale quantum-currency-api=3
   ```

### Option 3: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster 1.20+
- kubectl CLI

#### Steps

1. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f k8s/quantum-currency-deployment.yaml
   ```

2. **Check deployment status:**
   ```bash
   kubectl get pods -l app=quantum-currency
   kubectl get services
   ```

## 🔐 Security Hardening

### SSL/TLS Configuration

The Nginx configuration uses TLS 1.3 only with A+ rated ciphers:

```nginx
ssl_protocols TLSv1.3;
ssl_prefer_server_ciphers off;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
```

### User Permissions

- Dedicated service user (`quantum_user`) with minimal privileges
- Socket permissions restricted to `quantum_user:www-data`
- Application files owned by `quantum_user`

### Security Headers

Nginx adds security headers to all responses:
- `X-Frame-Options: SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: no-referrer-when-downgrade`
- `Content-Security-Policy`

## ⚡ High Availability Features

### Blue/Green Deployment

1. Maintain two identical environments (Blue=Live, Green=Staging)
2. Deploy new code to Green environment
3. After verification, switch Load Balancer traffic to Green

### Gunicorn Graceful Reload

Use Systemd to reload workers without dropping active connections:
```bash
sudo systemctl reload quantum-gunicorn
```

### Horizontal Scaling

- Multiple Gunicorn workers based on CPU cores
- Kubernetes Horizontal Pod Autoscaler
- Load balancing across multiple instances

## 🔭 Observability and Auto-Healing

### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:
- `quantum_currency_lambda_t` - Dynamic Lambda (λ(t)) value
- `quantum_currency_c_t` - Coherence Density (Ĉ(t)) value
- `quantum_currency_active_connections` - Active connections
- `quantum_currency_uptime` - System uptime
- `quantum_currency_memory_usage_mb` - Memory usage
- `quantum_currency_cpu_usage_percent` - CPU usage

### Alerting Rules

Prometheus alerting rules defined in `prometheus/rules/quantum_currency_alerts.yml`:
- **CoherenceDensityCritical** - Triggers when Ĉ(t) < 0.85
- **LambdaDriftWarning** - Triggers when λ(t) is out of bounds [0.8, 1.2]
- **HighCPUUsage** - Triggers when CPU usage > 80%
- **HighMemoryUsage** - Triggers when memory usage > 500MB
- **ServiceDown** - Triggers when service is not responding

### Auto-Healing Actions

Systemd timer runs `scripts/healing_script.sh` every 5 minutes:

1. **Critical Failure** (Ĉ(t) < 0.85): Hard restart service
2. **Warning Drift** (λ(t) out of bounds): Soft recalibration via CLI tool

## 📋 Monitoring Endpoints

- **Health Check**: `http://localhost/health`
- **Metrics**: `http://localhost/metrics`
- **Dashboard**: `http://localhost:3000` (when Grafana is deployed)

## 🛠️ Maintenance Operations

### Restart Service
```bash
sudo systemctl restart quantum-gunicorn
```

### View Logs
```bash
sudo journalctl -u quantum-gunicorn -f
```

### Check Service Status
```bash
sudo systemctl status quantum-gunicorn
```

### Scale Horizontally (Docker)
```bash
docker-compose -f docker-compose.production.yml up -d --scale quantum-currency-api=5
```

### Scale Horizontally (Kubernetes)
```bash
kubectl scale deployment quantum-currency-api --replicas=5
```

## 📊 Performance Tuning

### Gunicorn Configuration
- Workers: Number of CPU cores (`$(nproc)`)
- Threads: 2 per worker
- Timeout: 90 seconds

### Nginx Configuration
- Gzip compression enabled
- Connection timeouts optimized
- Security headers added

### Database Optimization
- Connection pooling
- Query optimization
- Indexing strategies

## 🆘 Troubleshooting

### Service Won't Start
1. Check logs: `sudo journalctl -u quantum-gunicorn`
2. Verify dependencies: `pip3 list | grep -E "(flask|gunicorn)"`
3. Check permissions: `ls -la /opt/quantum-currency`

### Health Checks Failing
1. Check application logs
2. Verify database connectivity
3. Check resource usage (CPU, memory)

### High Latency
1. Check system resources
2. Analyze slow queries
3. Optimize network configuration

## 🔄 Backup and Recovery

### Database Backup
```bash
pg_dump quantum_currency > quantum_currency_backup.sql
```

### Configuration Backup
```bash
tar -czf quantum_currency_config_backup.tar.gz /etc/systemd/system/quantum-gunicorn.service /etc/nginx/sites-available/quantum-currency.conf
```

### Application Backup
```bash
tar -czf quantum_currency_app_backup.tar.gz /opt/quantum-currency
```

## 📈 Performance Benchmarks

### Expected Performance
- Response time: < 100ms for 95% of requests
- Throughput: 1000+ requests/second
- Memory usage: < 500MB per instance
- CPU usage: < 50% under normal load

### Load Testing
Use tools like Apache Bench or Locust:
```bash
ab -n 10000 -c 100 http://localhost/health
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Systemd Documentation](https://www.freedesktop.org/wiki/Software/systemd/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---
*Quantum Currency Production Deployment Guide v1.0*