# Guía de Despliegue AEGIS

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Preparación del Entorno](#preparación-del-entorno)
4. [Despliegue Local](#despliegue-local)
5. [Despliegue en Docker](#despliegue-en-docker)
6. [Despliegue en Kubernetes](#despliegue-en-kubernetes)
7. [Despliegue en la Nube](#despliegue-en-la-nube)
8. [Configuración de Producción](#configuración-de-producción)
9. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)
10. [Resolución de Problemas](#resolución-de-problemas)

## Introducción

Esta guía proporciona instrucciones detalladas para desplegar AEGIS en diferentes entornos, desde desarrollo local hasta producción en la nube. AEGIS está diseñado para ser altamente configurable y adaptable a diversos escenarios de despliegue.

### Arquitectura de Despliegue

AEGIS sigue una arquitectura modular que permite despliegues flexibles:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Core Node     │    │   Storage       │
│   (Opcional)    │◄──►│   (Requerido)   │◄──►│   (Requerido)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   P2P Network   │
                       │   (Automático)  │
                       └─────────────────┘
```

## Requisitos del Sistema

### Requisitos Mínimos

#### Hardware
- **CPU**: 2 cores (x86_64 o ARM64)
- **RAM**: 4 GB
- **Almacenamiento**: 20 GB SSD
- **Red**: 10 Mbps bidireccional

#### Software
- **Sistema Operativo**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows (10+)
- **Python**: 3.9+
- **Dependencias**: Ver `requirements.txt`

### Requisitos Recomendados para Producción

#### Hardware
- **CPU**: 8+ cores (x86_64)
- **RAM**: 16+ GB
- **Almacenamiento**: 100+ GB NVMe SSD
- **Red**: 100+ Mbps bidireccional con baja latencia

#### Software
- **Sistema Operativo**: Ubuntu 22.04 LTS o CentOS 8+
- **Python**: 3.11+
- **Base de Datos**: PostgreSQL 14+ (para almacenamiento persistente)
- **Proxy Reverso**: Nginx o HAProxy
- **Monitoreo**: Prometheus + Grafana

## Preparación del Entorno

### 1. Instalación de Dependencias del Sistema

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
                    build-essential libssl-dev libffi-dev \
                    postgresql-client redis-tools git curl
```

#### CentOS/RHEL
```bash
sudo dnf update
sudo dnf install -y python3.11 python3.11-devel python3.11-pip \
                    gcc openssl-devel libffi-devel \
                    postgresql redis git curl
```

#### macOS
```bash
# Instalar Homebrew si no está instalado
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar dependencias
brew install python@3.11 postgresql redis git
```

### 2. Configuración del Entorno Python

```bash
# Crear entorno virtual
python3.11 -m venv aegis-env
source aegis-env/bin/activate  # Linux/macOS
# aegis-env\Scripts\activate  # Windows

# Actualizar pip
pip install --upgrade pip setuptools wheel

# Instalar AEGIS
pip install -r requirements.txt
```

### 3. Configuración de Variables de Entorno

Crear archivo `.env`:

```bash
# Configuración básica
AEGIS_NODE_ID=node-$(uuidgen)
AEGIS_ENVIRONMENT=production
AEGIS_LOG_LEVEL=INFO

# Red P2P
AEGIS_P2P_PORT=8000
AEGIS_P2P_HOST=0.0.0.0
AEGIS_BOOTSTRAP_NODES=node1.aegis.network:8000,node2.aegis.network:8000

# Almacenamiento
AEGIS_STORAGE_TYPE=postgresql
AEGIS_DATABASE_URL=postgresql://user:pass@localhost:5432/aegis

# Seguridad
AEGIS_ENCRYPTION_KEY=$(openssl rand -base64 32)
AEGIS_JWT_SECRET=$(openssl rand -base64 64)

# Monitoreo
AEGIS_METRICS_ENABLED=true
AEGIS_METRICS_PORT=9090
```

## Despliegue Local

### Despliegue de Desarrollo

Para desarrollo y pruebas locales:

```bash
# 1. Clonar repositorio
git clone https://github.com/aegis-project/aegis.git
cd aegis

# 2. Configurar entorno
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configuración mínima
cp config/development.yaml config/local.yaml
# Editar config/local.yaml según necesidades

# 4. Inicializar base de datos local
python scripts/init_db.py

# 5. Ejecutar nodo
python main.py --config config/local.yaml
```

### Despliegue de Pruebas

Para un entorno de pruebas más robusto:

```bash
# 1. Configurar PostgreSQL local
sudo -u postgres createdb aegis_test
sudo -u postgres psql -c "CREATE USER aegis WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aegis_test TO aegis;"

# 2. Configurar Redis (opcional, para caché)
sudo systemctl start redis
sudo systemctl enable redis

# 3. Configuración de pruebas
cat > config/testing.yaml << EOF
node:
  id: "test-node-1"
  environment: "testing"

storage:
  type: "postgresql"
  connection_string: "postgresql://aegis:secure_password@localhost:5432/aegis_test"

p2p:
  port: 8001
  bootstrap_nodes: []

logging:
  level: "DEBUG"
  file: "logs/aegis-test.log"
EOF

# 4. Ejecutar con configuración de pruebas
python main.py --config config/testing.yaml
```

## Despliegue en Docker

### Dockerfile Optimizado

```dockerfile
# Multi-stage build para optimizar tamaño
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Etapa de producción
FROM python:3.11-slim

# Crear usuario no-root
RUN groupadd -r aegis && useradd -r -g aegis aegis

# Instalar dependencias de runtime
RUN apt-get update && apt-get install -y \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias de Python
COPY --from=builder /root/.local /home/aegis/.local

# Configurar PATH
ENV PATH=/home/aegis/.local/bin:$PATH

WORKDIR /app

# Copiar código fuente
COPY --chown=aegis:aegis . .

# Cambiar a usuario no-root
USER aegis

# Exponer puertos
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Comando por defecto
CMD ["python", "main.py"]
```

### Docker Compose para Desarrollo

```yaml
version: '3.8'

services:
  aegis-node:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - AEGIS_ENVIRONMENT=development
      - AEGIS_DATABASE_URL=postgresql://aegis:password@postgres:5432/aegis
      - AEGIS_REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    networks:
      - aegis-network

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: aegis
      POSTGRES_USER: aegis
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - aegis-network

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - aegis-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - aegis-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - aegis-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  aegis-network:
    driver: bridge
```

### Docker Compose para Producción

```yaml
version: '3.8'

services:
  aegis-node:
    image: aegis:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - AEGIS_ENVIRONMENT=production
      - AEGIS_DATABASE_URL=${DATABASE_URL}
      - AEGIS_ENCRYPTION_KEY=${ENCRYPTION_KEY}
    env_file:
      - .env.production
    volumes:
      - ./config/production.yaml:/app/config/production.yaml:ro
      - aegis_logs:/app/logs
    networks:
      - aegis-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - aegis_logs:/var/log/nginx
    depends_on:
      - aegis-node
    networks:
      - aegis-network

volumes:
  aegis_logs:

networks:
  aegis-network:
    driver: bridge
```

## Despliegue en Kubernetes

### Namespace y ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aegis-system
  labels:
    name: aegis-system

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aegis-config
  namespace: aegis-system
data:
  production.yaml: |
    node:
      id: "${AEGIS_NODE_ID}"
      environment: "production"
    
    p2p:
      port: 8000
      host: "0.0.0.0"
      bootstrap_nodes:
        - "aegis-node-0.aegis-headless.aegis-system.svc.cluster.local:8000"
        - "aegis-node-1.aegis-headless.aegis-system.svc.cluster.local:8000"
    
    storage:
      type: "postgresql"
      connection_string: "${DATABASE_URL}"
    
    monitoring:
      enabled: true
      metrics_port: 9090
    
    logging:
      level: "INFO"
      format: "json"
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: aegis-secrets
  namespace: aegis-system
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  encryption-key: <base64-encoded-encryption-key>
  jwt-secret: <base64-encoded-jwt-secret>
```

### StatefulSet para Nodos AEGIS

```yaml
# statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: aegis-node
  namespace: aegis-system
spec:
  serviceName: aegis-headless
  replicas: 3
  selector:
    matchLabels:
      app: aegis-node
  template:
    metadata:
      labels:
        app: aegis-node
    spec:
      containers:
      - name: aegis
        image: aegis:latest
        ports:
        - containerPort: 8000
          name: p2p
        - containerPort: 9090
          name: metrics
        env:
        - name: AEGIS_NODE_ID
          value: "$(hostname)"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aegis-secrets
              key: database-url
        - name: AEGIS_ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: aegis-secrets
              key: encryption-key
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: aegis-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Services

```yaml
# services.yaml
apiVersion: v1
kind: Service
metadata:
  name: aegis-headless
  namespace: aegis-system
spec:
  clusterIP: None
  selector:
    app: aegis-node
  ports:
  - port: 8000
    name: p2p

---
apiVersion: v1
kind: Service
metadata:
  name: aegis-service
  namespace: aegis-system
spec:
  selector:
    app: aegis-node
  ports:
  - port: 8000
    targetPort: 8000
    name: p2p
  - port: 9090
    targetPort: 9090
    name: metrics
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: aegis-metrics
  namespace: aegis-system
  labels:
    app: aegis-node
spec:
  selector:
    app: aegis-node
  ports:
  - port: 9090
    targetPort: 9090
    name: metrics
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aegis-ingress
  namespace: aegis-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
spec:
  tls:
  - hosts:
    - aegis.yourdomain.com
    secretName: aegis-tls
  rules:
  - host: aegis.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aegis-service
            port:
              number: 8000
```

### Despliegue en Kubernetes

```bash
# 1. Crear namespace y configuración
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# 2. Desplegar aplicación
kubectl apply -f statefulset.yaml
kubectl apply -f services.yaml
kubectl apply -f ingress.yaml

# 3. Verificar despliegue
kubectl get pods -n aegis-system
kubectl get services -n aegis-system
kubectl logs -f aegis-node-0 -n aegis-system

# 4. Escalar nodos
kubectl scale statefulset aegis-node --replicas=5 -n aegis-system
```

## Despliegue en la Nube

### AWS (Amazon Web Services)

#### Usando ECS (Elastic Container Service)

```json
{
  "family": "aegis-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/aegisTaskRole",
  "containerDefinitions": [
    {
      "name": "aegis-node",
      "image": "your-account.dkr.ecr.region.amazonaws.com/aegis:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        },
        {
          "containerPort": 9090,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AEGIS_ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:aegis/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/aegis",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python scripts/health_check.py || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Terraform para AWS

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC y Networking
resource "aws_vpc" "aegis_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "aegis-vpc"
  }
}

resource "aws_subnet" "aegis_subnet" {
  count             = 2
  vpc_id            = aws_vpc.aegis_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "aegis-subnet-${count.index + 1}"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "aegis_cluster" {
  name = "aegis-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "aegis_alb" {
  name               = "aegis-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.aegis_alb_sg.id]
  subnets            = aws_subnet.aegis_subnet[*].id

  enable_deletion_protection = false
}

# RDS para PostgreSQL
resource "aws_db_instance" "aegis_db" {
  identifier     = "aegis-db"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "aegis"
  username = "aegis"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.aegis_db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.aegis_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
}
```

### Google Cloud Platform (GCP)

#### Usando Google Kubernetes Engine (GKE)

```bash
# 1. Crear cluster GKE
gcloud container clusters create aegis-cluster \
    --zone=us-central1-a \
    --num-nodes=3 \
    --machine-type=e2-standard-2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10 \
    --enable-autorepair \
    --enable-autoupgrade

# 2. Configurar kubectl
gcloud container clusters get-credentials aegis-cluster --zone=us-central1-a

# 3. Crear base de datos Cloud SQL
gcloud sql instances create aegis-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1

# 4. Desplegar aplicación
kubectl apply -f k8s/
```

### Microsoft Azure

#### Usando Azure Container Instances (ACI)

```yaml
# azure-container-group.yaml
apiVersion: 2019-12-01
location: eastus
name: aegis-container-group
properties:
  containers:
  - name: aegis-node
    properties:
      image: aegis:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
      - port: 9090
        protocol: TCP
      environmentVariables:
      - name: AEGIS_ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: postgresql://user:pass@server:5432/aegis
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 9090
tags:
  environment: production
  application: aegis
type: Microsoft.ContainerInstance/containerGroups
```

## Configuración de Producción

### Configuración de Seguridad

```yaml
# config/production.yaml
security:
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_interval: 86400  # 24 horas
  
  authentication:
    jwt:
      algorithm: "RS256"
      expiration: 3600  # 1 hora
      refresh_expiration: 604800  # 7 días
  
  network:
    tls:
      enabled: true
      cert_file: "/etc/ssl/certs/aegis.crt"
      key_file: "/etc/ssl/private/aegis.key"
      min_version: "1.3"
    
    rate_limiting:
      enabled: true
      requests_per_minute: 1000
      burst_size: 100

logging:
  level: "INFO"
  format: "json"
  output: "file"
  file: "/var/log/aegis/aegis.log"
  rotation:
    max_size: "100MB"
    max_files: 10
    max_age: 30

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_port: 8080
  
  prometheus:
    enabled: true
    scrape_interval: "15s"
  
  alerts:
    enabled: true
    webhook_url: "https://alerts.yourdomain.com/webhook"

performance:
  max_connections: 1000
  connection_timeout: 30
  read_timeout: 60
  write_timeout: 60
  
  cache:
    enabled: true
    type: "redis"
    url: "redis://redis-cluster:6379"
    ttl: 3600

storage:
  type: "postgresql"
  connection_string: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  
  backup:
    enabled: true
    interval: 3600  # 1 hora
    retention: 168  # 7 días
    storage_type: "s3"
    s3_bucket: "aegis-backups"
```

### Configuración de Nginx

```nginx
# /etc/nginx/sites-available/aegis
upstream aegis_backend {
    least_conn;
    server aegis-node-1:8000 max_fails=3 fail_timeout=30s;
    server aegis-node-2:8000 max_fails=3 fail_timeout=30s;
    server aegis-node-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name aegis.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aegis.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/aegis.crt;
    ssl_certificate_key /etc/ssl/private/aegis.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy Configuration
    location / {
        proxy_pass http://aegis_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health Check Endpoint
    location /health {
        access_log off;
        proxy_pass http://aegis_backend/health;
        proxy_set_header Host $host;
    }

    # Metrics Endpoint (restrict access)
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        
        proxy_pass http://aegis_backend:9090/metrics;
        proxy_set_header Host $host;
    }

    # Static files (if any)
    location /static/ {
        alias /var/www/aegis/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Logging
    access_log /var/log/nginx/aegis_access.log;
    error_log /var/log/nginx/aegis_error.log;
}
```

### Configuración de Systemd

```ini
# /etc/systemd/system/aegis.service
[Unit]
Description=AEGIS Distributed System Node
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=aegis
Group=aegis
WorkingDirectory=/opt/aegis
Environment=PYTHONPATH=/opt/aegis
EnvironmentFile=/opt/aegis/.env
ExecStart=/opt/aegis/venv/bin/python main.py --config /opt/aegis/config/production.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aegis

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/aegis/logs /opt/aegis/data
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=2G
CPUQuota=100%

[Install]
WantedBy=multi-user.target
```

## Monitoreo y Mantenimiento

### Configuración de Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "aegis_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'aegis-nodes'
    static_configs:
      - targets: 
        - 'aegis-node-1:9090'
        - 'aegis-node-2:9090'
        - 'aegis-node-3:9090'
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets:
        - 'node-1:9100'
        - 'node-2:9100'
        - 'node-3:9100'
```

### Reglas de Alertas

```yaml
# aegis_rules.yml
groups:
- name: aegis.rules
  rules:
  - alert: AegisNodeDown
    expr: up{job="aegis-nodes"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "AEGIS node is down"
      description: "AEGIS node {{ $labels.instance }} has been down for more than 1 minute."

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 90% on {{ $labels.instance }}"

  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 80% on {{ $labels.instance }}"

  - alert: P2PConnectionsLow
    expr: aegis_p2p_connected_peers < 3
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Low P2P connections"
      description: "Node {{ $labels.instance }} has less than 3 P2P connections"

  - alert: ConsensusFailure
    expr: increase(aegis_consensus_failures_total[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Consensus failure detected"
      description: "Consensus failures detected on {{ $labels.instance }}"
```

### Scripts de Mantenimiento

```bash
#!/bin/bash
# scripts/maintenance.sh

set -euo pipefail

AEGIS_HOME="/opt/aegis"
LOG_FILE="/var/log/aegis/maintenance.log"
BACKUP_DIR="/backup/aegis"
RETENTION_DAYS=7

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Función de backup
backup_data() {
    log "Starting backup process..."
    
    # Crear directorio de backup con timestamp
    BACKUP_PATH="$BACKUP_DIR/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$BACKUP_PATH"
    
    # Backup de configuración
    cp -r "$AEGIS_HOME/config" "$BACKUP_PATH/"
    
    # Backup de datos (si existe directorio local)
    if [ -d "$AEGIS_HOME/data" ]; then
        cp -r "$AEGIS_HOME/data" "$BACKUP_PATH/"
    fi
    
    # Backup de base de datos
    if [ -n "${DATABASE_URL:-}" ]; then
        pg_dump "$DATABASE_URL" > "$BACKUP_PATH/database.sql"
    fi
    
    # Comprimir backup
    tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "$(basename "$BACKUP_PATH")"
    rm -rf "$BACKUP_PATH"
    
    log "Backup completed: $BACKUP_PATH.tar.gz"
}

# Función de limpieza de logs
cleanup_logs() {
    log "Cleaning up old logs..."
    
    find "$AEGIS_HOME/logs" -name "*.log" -mtime +$RETENTION_DAYS -delete
    find "/var/log/aegis" -name "*.log" -mtime +$RETENTION_DAYS -delete
    
    log "Log cleanup completed"
}

# Función de limpieza de backups antiguos
cleanup_backups() {
    log "Cleaning up old backups..."
    
    find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    
    log "Backup cleanup completed"
}

# Función de verificación de salud
health_check() {
    log "Performing health check..."
    
    # Verificar que el servicio esté corriendo
    if ! systemctl is-active --quiet aegis; then
        log "ERROR: AEGIS service is not running"
        return 1
    fi
    
    # Verificar conectividad de base de datos
    if [ -n "${DATABASE_URL:-}" ]; then
        if ! pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; then
            log "ERROR: Database is not accessible"
            return 1
        fi
    fi
    
    # Verificar endpoint de salud
    if ! curl -f http://localhost:8080/health >/dev/null 2>&1; then
        log "ERROR: Health endpoint is not responding"
        return 1
    fi
    
    log "Health check passed"
    return 0
}

# Función de actualización
update_system() {
    log "Starting system update..."
    
    # Detener servicio
    systemctl stop aegis
    
    # Backup antes de actualizar
    backup_data
    
    # Actualizar código (ejemplo con git)
    cd "$AEGIS_HOME"
    git pull origin main
    
    # Actualizar dependencias
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Ejecutar migraciones si existen
    if [ -f "scripts/migrate.py" ]; then
        python scripts/migrate.py
    fi
    
    # Reiniciar servicio
    systemctl start aegis
    
    # Verificar que todo funcione
    sleep 10
    if health_check; then
        log "Update completed successfully"
    else
        log "ERROR: Update failed, consider rollback"
        return 1
    fi
}

# Función principal
main() {
    case "${1:-}" in
        backup)
            backup_data
            ;;
        cleanup)
            cleanup_logs
            cleanup_backups
            ;;
        health)
            health_check
            ;;
        update)
            update_system
            ;;
        daily)
            backup_data
            cleanup_logs
            cleanup_backups
            health_check
            ;;
        *)
            echo "Usage: $0 {backup|cleanup|health|update|daily}"
            exit 1
            ;;
    esac
}

main "$@"
```

### Crontab para Mantenimiento Automático

```bash
# Crontab para usuario aegis
# Editar con: crontab -e

# Backup diario a las 2:00 AM
0 2 * * * /opt/aegis/scripts/maintenance.sh backup

# Limpieza semanal los domingos a las 3:00 AM
0 3 * * 0 /opt/aegis/scripts/maintenance.sh cleanup

# Health check cada 5 minutos
*/5 * * * * /opt/aegis/scripts/maintenance.sh health

# Rotación de logs diaria a las 1:00 AM
0 1 * * * /usr/sbin/logrotate /etc/logrotate.d/aegis
```

## Resolución de Problemas

### Problemas Comunes y Soluciones

#### 1. Nodo no puede conectarse a la red P2P

**Síntomas:**
- Logs muestran "No peers available"
- Métricas indican 0 conexiones P2P

**Diagnóstico:**
```bash
# Verificar conectividad de red
telnet bootstrap-node.aegis.network 8000

# Verificar configuración de firewall
sudo ufw status
sudo iptables -L

# Verificar logs de P2P
grep "p2p" /var/log/aegis/aegis.log
```

**Soluciones:**
```bash
# 1. Verificar configuración de bootstrap nodes
cat config/production.yaml | grep -A 5 bootstrap_nodes

# 2. Abrir puertos en firewall
sudo ufw allow 8000/tcp

# 3. Verificar NAT/Port forwarding si está detrás de router
# 4. Reiniciar servicio P2P
systemctl restart aegis
```

#### 2. Alta latencia en operaciones de base de datos

**Síntomas:**
- Timeouts en operaciones de storage
- Métricas muestran alta latencia de DB

**Diagnóstico:**
```bash
# Verificar conexiones activas
psql "$DATABASE_URL" -c "SELECT count(*) FROM pg_stat_activity;"

# Verificar queries lentas
psql "$DATABASE_URL" -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Verificar locks
psql "$DATABASE_URL" -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Soluciones:**
```bash
# 1. Optimizar pool de conexiones
# Editar config/production.yaml:
# storage:
#   pool_size: 10
#   max_overflow: 20

# 2. Añadir índices faltantes
psql "$DATABASE_URL" -c "CREATE INDEX CONCURRENTLY idx_table_column ON table(column);"

# 3. Analizar y optimizar queries
psql "$DATABASE_URL" -c "ANALYZE;"
```

#### 3. Consumo excesivo de memoria

**Síntomas:**
- OOM kills en logs del sistema
- Métricas muestran uso de memoria > 90%

**Diagnóstico:**
```bash
# Verificar uso de memoria del proceso
ps aux | grep aegis

# Verificar memory leaks
valgrind --tool=memcheck --leak-check=full python main.py

# Verificar configuración de límites
systemctl show aegis | grep Memory
```

**Soluciones:**
```bash
# 1. Ajustar límites de memoria en systemd
sudo systemctl edit aegis
# Añadir:
# [Service]
# MemoryMax=1G

# 2. Optimizar configuración de cache
# En config/production.yaml:
# performance:
#   cache:
#     ttl: 1800  # Reducir TTL
#     max_size: 1000  # Limitar tamaño

# 3. Habilitar garbage collection más agresivo
export PYTHONHASHSEED=0
export PYTHONOPTIMIZE=1
```

#### 4. Fallos de consenso

**Síntomas:**
- Alertas de "ConsensusFailure"
- Nodos fuera de sincronización

**Diagnóstico:**
```bash
# Verificar estado de consenso
curl http://localhost:9090/metrics | grep consensus

# Verificar sincronización entre nodos
for node in node1 node2 node3; do
    echo "Node $node:"
    curl -s http://$node:9090/status | jq .block_height
done

# Verificar logs de consenso
grep "consensus" /var/log/aegis/aegis.log | tail -50
```

**Soluciones:**
```bash
# 1. Verificar conectividad entre nodos
for node in node1 node2 node3; do
    telnet $node 8000
done

# 2. Reiniciar nodos problemáticos
systemctl restart aegis

# 3. Verificar configuración de timeouts
# En config/production.yaml:
# consensus:
#   timeout: 30
#   retry_interval: 5
```

### Herramientas de Diagnóstico

#### Script de Diagnóstico Automático

```bash
#!/bin/bash
# scripts/diagnose.sh

AEGIS_HOME="/opt/aegis"
REPORT_FILE="/tmp/aegis_diagnostic_$(date +%Y%m%d_%H%M%S).txt"

echo "AEGIS Diagnostic Report - $(date)" > "$REPORT_FILE"
echo "=================================" >> "$REPORT_FILE"

# Información del sistema
echo -e "\n## System Information" >> "$REPORT_FILE"
uname -a >> "$REPORT_FILE"
cat /etc/os-release >> "$REPORT_FILE"

# Estado del servicio
echo -e "\n## Service Status" >> "$REPORT_FILE"
systemctl status aegis >> "$REPORT_FILE" 2>&1

# Uso de recursos
echo -e "\n## Resource Usage" >> "$REPORT_FILE"
ps aux | grep aegis >> "$REPORT_FILE"
free -h >> "$REPORT_FILE"
df -h >> "$REPORT_FILE"

# Conectividad de red
echo -e "\n## Network Connectivity" >> "$REPORT_FILE"
netstat -tlnp | grep :8000 >> "$REPORT_FILE"
ss -tlnp | grep :8000 >> "$REPORT_FILE"

# Logs recientes
echo -e "\n## Recent Logs (last 50 lines)" >> "$REPORT_FILE"
tail -50 /var/log/aegis/aegis.log >> "$REPORT_FILE"

# Configuración
echo -e "\n## Configuration" >> "$REPORT_FILE"
cat "$AEGIS_HOME/config/production.yaml" >> "$REPORT_FILE"

# Métricas de salud
echo -e "\n## Health Metrics" >> "$REPORT_FILE"
curl -s http://localhost:9090/metrics | grep -E "(aegis_|up|process_)" >> "$REPORT_FILE" 2>&1

echo "Diagnostic report generated: $REPORT_FILE"
```

## Proxy IPFS para SecureChat en producción

Para exponer la API de IPFS a la UI sin problemas de CORS y con buenas prácticas de seguridad, utiliza un reverse proxy en `nginx` que mapee el prefijo `/ipfs-api` hacia el API de IPFS (`http://localhost:5001`). Esto mantiene el mismo origen para la UI y evita bloqueos del navegador.

Ejemplo básico de configuración en `nginx`:

```
server {
    listen 80;
    server_name securechat.tu-dominio.com;

    # UI de SecureChat (build estático de Vite)
    root /var/www/secure-chat-ui;
    index index.html;

    location / {
        try_files $uri /index.html;
    }

    # Proxy IPFS API
    location /ipfs-api/ {
        proxy_pass http://127.0.0.1:5001/; # API de IPFS local
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Opcional: restringir métodos
        # if ($request_method !~ ^(GET|POST|PUT|OPTIONS)$ ) { return 405; }
    }
}
```

Notas y seguridad:
- La UI debe apuntar a `/ipfs-api` (ya se usa en `src/App.jsx`).
- Si usas HTTPS, configura `listen 443 ssl http2;` y certificados.
- Si IPFS corre en otro host, ajusta `proxy_pass` y abre firewall con reglas mínimas.
- Revisa logs de `nginx` e IPFS para actividad inusual y limita endpoints sensibles si tu caso lo requiere.

### Contacto y Soporte

Para soporte técnico y resolución de problemas:

- **Documentación**: https://docs.aegis-project.org
- **Issues**: https://github.com/aegis-project/aegis/issues
- **Comunidad**: https://community.aegis-project.org
- **Soporte Enterprise**: support@aegis-project.org

---

*Esta guía de despliegue se actualiza regularmente. Para la versión más reciente, consulte la documentación oficial en línea.*