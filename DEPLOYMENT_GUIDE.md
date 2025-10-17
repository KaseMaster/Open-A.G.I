# 🚀 Guía de Despliegue AEGIS Framework

## Descripción

Esta guía proporciona instrucciones detalladas para desplegar AEGIS Framework en diferentes entornos usando Docker Compose y configuración automatizada.

## 📋 Prerrequisitos

- **Docker** y **Docker Compose** (versión 3.8+)
- **Python 3.9+** (para desarrollo local)
- **TOR** (opcional, para funcionalidades avanzadas)
- **4GB+ RAM** recomendados
- **Conexión a internet** estable

## 🐳 Despliegue con Docker Compose (Recomendado)

### Instalación Rápida

1. **Clonar el repositorio:**
```bash
git clone https://github.com/KaseMaster/Open-A.G.I.git
cd Open-A.G.I
```

2. **Ejecutar script de despliegue:**
```bash
# En Linux/macOS
chmod +x deploy.sh
./deploy.sh

# En Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File deploy.ps1
```

3. **O despliegue directo con Docker Compose:**
```bash
# Construir e iniciar todos los servicios
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Ver estado de servicios
docker-compose ps
```

### Servicios Incluidos

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| **aegis-node** | 8080 | Nodo principal del framework |
| **web-dashboard** | 8051 | Dashboard web independiente |
| **tor** | 9050/9051 | Servicio TOR para anonimato |
| **redis** | 6379 | Cache y sesiones |
| **nginx** | 80/443 | Reverse proxy y SSL |
| **monitoring** | 9090 | Métricas Prometheus |

### URLs de Acceso

- **Dashboard Principal:** http://localhost:8080
- **Dashboard Web:** http://localhost:8051
- **Métricas:** http://localhost:9090
- **API REST:** http://localhost:8081

## 🛠️ Configuración Manual

### 1. Entorno Virtual Python

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno (Linux/macOS)
source venv/bin/activate

# Activar entorno (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración de TOR

```bash
# Instalar TOR (Ubuntu/Debian)
sudo apt-get install tor

# Iniciar servicio TOR
sudo systemctl start tor
sudo systemctl enable tor

# Configurar permisos (opcional)
sudo usermod -a -G debian-tor $USER
```

### 3. Variables de Entorno

Crear archivo `.env`:

```bash
cat > .env << EOF
# Configuración de Red
TOR_CONTROL_PORT=9051
TOR_SOCKS_PORT=9050
P2P_PORT=8080

# Configuración de Seguridad
SECURITY_LEVEL=HIGH
MIN_COMPUTATION_SCORE=50.0
BYZANTINE_THRESHOLD_RATIO=0.33

# Configuración de Consenso
POC_INTERVAL=300
PBFT_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=distributed_ai.log

# Dashboard
AEGIS_DASHBOARD_PORT=8080
EOF
```

## 🏃‍♂️ Inicio de Servicios

### Iniciar Nodo Individual

```bash
# Health check del sistema
python main.py health-check

# Listar módulos disponibles
python main.py list-modules

# Iniciar nodo completo
python main.py start-node

# Iniciar solo el dashboard
python main.py start-dashboard --port 8080
```

### Comandos CLI Disponibles

```bash
# Iniciar nodo completo
python main.py start-node [--dry-run] [--config path/to/config.json]

# Verificar salud del sistema
python main.py health-check

# Listar módulos disponibles
python main.py list-modules

# Iniciar dashboard específico
python main.py start-dashboard --type monitoring --host 0.0.0.0 --port 8080
```

## 🔍 Verificación de Instalación

### Tests Automatizados

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_consensus.py -v
python -m pytest tests/test_security.py -v
python -m pytest tests/test_tor_integration.py -v
```

### Health Check Manual

```bash
# Verificar módulos Python
python main.py health-check

# Verificar conectividad TOR
curl -s http://localhost:9051/tor/status

# Verificar Redis
redis-cli ping

# Verificar servicios Docker
docker-compose ps
docker-compose logs [service-name]
```

## 🔧 Configuración Avanzada

### Configuración Multi-Entorno

Crear archivos de configuración específicos:

```bash
# Desarrollo
cp config/app_config.json config/app_config.dev.json

# Producción
cp config/app_config.json config/app_config.prod.json

# Staging
cp config/app_config.json config/app_config.staging.json
```

### Configuración de Nginx

El archivo `nginx.conf` incluye configuración para:

- **Reverse proxy** a servicios backend
- **SSL/TLS** con certificados auto-firmados
- **Rate limiting** y seguridad
- **Compresión** gzip
- **Caching** estático

### Monitoreo y Logs

```bash
# Ver logs de todos los servicios
docker-compose logs -f

# Logs de servicio específico
docker-compose logs -f aegis-node

# Logs del sistema
tail -f logs/distributed_ai.log

# Métricas Prometheus
curl http://localhost:9090/metrics
```

## 🐛 Troubleshooting

### Problemas Comunes

1. **TOR no inicia:**
```bash
sudo systemctl status tor
sudo journalctl -u tor -f
```

2. **Puerto en uso:**
```bash
# Verificar puertos
netstat -tlnp | grep :8080
# Cambiar puerto en config/app_config.json
```

3. **Dependencias faltantes:**
```bash
pip install -r requirements.txt
# O con Docker
docker-compose build --no-cache
```

4. **Permisos de archivos:**
```bash
# Arreglar permisos
chmod +x scripts/*.sh
chmod 666 data/database/*.db
```

### Logs de Debug

```bash
# Activar logging verbose
export AEGIS_LOG_LEVEL=DEBUG

# Logs con timestamps
python main.py start-node 2>&1 | ts

# Logs a archivo
python main.py start-node > logs/aegis.log 2>&1
```

## 📊 Métricas y Monitoreo

### Prometheus Metrics

El sistema expone métricas en:

- **Nodos activos:** `aegis_active_nodes`
- **Estado de consenso:** `aegis_consensus_status`
- **Latencia de red:** `aegis_network_latency_ms`
- **Uso de recursos:** `aegis_resource_usage_percent`

### Health Endpoints

```bash
# Health check HTTP
curl http://localhost:8080/health

# Métricas detalladas
curl http://localhost:8080/metrics

# Estado de consenso
curl http://localhost:8080/consensus/status
```

## 🔒 Seguridad

### Configuración de Producción

1. **Cambiar puertos por defecto**
2. **Configurar SSL/TLS real**
3. **Configurar firewall**
4. **Rotar claves regularmente**
5. **Monitorear logs de seguridad**

```bash
# Configuración de firewall (Ubuntu/Debian)
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 9050
sudo ufw enable
```

### Mejores Prácticas

- **No ejecutar como root**
- **Validar todas las entradas**
- **Mantener dependencias actualizadas**
- **Rotar certificados regularmente**
- **Monitorear logs de seguridad**

## 📞 Soporte

Para problemas o preguntas:

1. **Verificar logs:** `docker-compose logs [service-name]`
2. **Health check:** `python main.py health-check`
3. **Tests:** `python -m pytest tests/ -v`
4. **Reportar issues:** GitHub Issues

---

**⚠️ IMPORTANTE:** Este framework es para investigación y desarrollo ético únicamente. El uso malicioso está estrictamente prohibido.
