# AEGIS - Sistema Avanzado de Gestión de Información y Seguridad

![AEGIS Logo](https://img.shields.io/badge/AEGIS-v2.0-blue?style=for-the-badge&logo=shield&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=flat-square)

## 🛡️ Descripción General

**AEGIS** (Analista Experto en Gestión de Información y Seguridad) es un sistema distribuido de alta seguridad diseñado para la gestión inteligente de información, análisis de vulnerabilidades y monitoreo en tiempo real. Combina tecnologías avanzadas de criptografía, redes P2P, consenso distribuido y análisis predictivo.

### 🎯 Características Principales

- **🔐 Criptografía Avanzada**: Implementación de algoritmos post-cuánticos y rotación automática de claves
- **🌐 Red P2P Segura**: Comunicación descentralizada con autenticación robusta
- **🤝 Consenso Distribuido**: Algoritmo híbrido PoC+PBFT para integridad de datos
- **📊 Monitoreo Inteligente**: Dashboard en tiempo real con análisis predictivo
- **🔄 Backup Automático**: Sistema de respaldo cifrado con soporte cloud
- **🧪 Testing Integral**: Framework completo de tests unitarios, integración y rendimiento
- **🚨 Alertas IA**: Sistema inteligente de detección de anomalías

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    AEGIS Core System                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Crypto Layer  │   P2P Network   │   Consensus Engine      │
│   - ChaCha20    │   - Zeroconf    │   - PoC + PBFT         │
│   - Blake3      │   - TOR Support │   - Byzantine Fault    │
│   - Post-Quantum│   - Mesh Topo   │   - State Machine      │
├─────────────────┼─────────────────┼─────────────────────────┤
│  Monitoring     │   Web Dashboard │   Alert System         │
│  - Real-time    │   - SocketIO    │   - AI Analysis        │
│  - Metrics      │   - Charts      │   - Multi-channel      │
│  - Performance  │   - Responsive  │   - Predictive         │
├─────────────────┼─────────────────┼─────────────────────────┤
│  Backup System  │   Test Framework│   API Server           │
│  - Encrypted    │   - Unit Tests  │   - REST + WebSocket   │
│  - Cloud Sync   │   - Integration │   - JWT Auth           │
│  - Automated    │   - Performance │   - Rate Limiting      │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## 🚀 Instalación Rápida

### Prerrequisitos

```bash
# Python 3.8 o superior
python --version

# Dependencias del sistema (Ubuntu/Debian)
sudo apt update
sudo apt install python3-pip python3-venv tor

# Dependencias del sistema (Windows)
# Instalar Python desde python.org
# Instalar Tor Browser o Tor Expert Bundle
```

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/aegis.git
cd aegis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones
```

### Configuración Inicial

```bash
# Generar configuración por defecto
python main.py --help

# Verificar módulos disponibles
python main.py list-modules

# Ejecutar verificación de salud
python main.py health-check

# Iniciar en modo dry-run (solo validación)
python main.py start-node --dry-run
```

## 🎮 Uso Básico

### Iniciar el Sistema Completo

```bash
# Iniciar todos los módulos
python main.py start-node

# Iniciar con configuración personalizada
python main.py start-node --config mi_config.json

# Modo desarrollo (logs detallados)
LOG_LEVEL=DEBUG python main.py start-node
```

### Acceso a Interfaces

- **Dashboard Web**: http://localhost:8080
- **API REST**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **Métricas**: http://localhost:8000/metrics

### Comandos Útiles

```bash
# Ejecutar tests
python test_suites/run_tests.py

# Tests específicos
python test_suites/run_tests.py --suite crypto
python test_suites/run_tests.py --suite integration

# Generar reportes
python test_suites/run_tests.py --report-format html

# Monitoreo en tiempo real
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/metrics
```

## 📋 Configuración Avanzada

### Archivo de Configuración

El sistema utiliza un archivo JSON para configuración completa:

```json
{
  "app": {
    "log_level": "INFO",
    "enable": {
      "tor": true,
      "p2p": true,
      "crypto": true,
      "consensus": true,
      "monitoring": true,
      "web_dashboard": true,
      "backup_system": true,
      "test_framework": true
    }
  },
  "crypto": {
    "rotate_interval_hours": 24,
    "hash": "blake3",
    "symmetric": "chacha20-poly1305"
  },
  "p2p": {
    "discovery": "zeroconf",
    "heartbeat_interval_sec": 30
  }
}
```

### Variables de Entorno

```bash
# Configuración básica
LOG_LEVEL=INFO
AEGIS_CONFIG_PATH=/path/to/config.json

# Configuración de red
P2P_PORT=8001
API_PORT=8000
DASHBOARD_PORT=8080

# Configuración de seguridad
CRYPTO_KEY_ROTATION_HOURS=24
BACKUP_ENCRYPTION_ENABLED=true

# Configuración TOR (opcional)
TOR_SOCKS_PORT=9050
TOR_CONTROL_PORT=9051
```

## 🔧 Desarrollo

### Estructura del Proyecto

```
aegis/
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias Python
├── .env.example           # Plantilla de configuración
├── docs/                  # Documentación
│   ├── README.md
│   ├── API.md
│   └── ARCHITECTURE.md
├── test_suites/           # Framework de testing
│   ├── __init__.py
│   ├── run_tests.py
│   ├── test_crypto.py
│   ├── test_p2p.py
│   ├── test_integration.py
│   └── test_performance.py
├── modules/               # Módulos del sistema
│   ├── crypto_framework.py
│   ├── p2p_network.py
│   ├── consensus_algorithm.py
│   ├── monitoring_dashboard.py
│   ├── web_dashboard.py
│   ├── backup_system.py
│   └── test_framework.py
└── logs/                  # Archivos de log
```

### Agregar Nuevos Módulos

1. **Crear el módulo**:
```python
# modules/mi_modulo.py
async def start_mi_modulo(config: dict):
    """Función de inicio del módulo"""
    pass

async def stop_mi_modulo():
    """Función de parada del módulo"""
    pass
```

2. **Registrar en main.py**:
```python
# Agregar importación segura
mi_mod, mi_err = safe_import("mi_modulo")

# Agregar verificación de errores
if mi_err:
    logger.warning(f"Mi módulo no disponible: {mi_err}")

# Agregar inicialización
if cfg["app"]["enable"].get("mi_modulo", False) and mi_mod:
    await module_call(mi_mod, "start_mi_modulo", cfg.get("mi_modulo", {}))
```

3. **Agregar configuración**:
```python
# En DEFAULT_CONFIG
"mi_modulo": {
    "enabled": True,
    "parametro1": "valor1",
    "parametro2": 42
}
```

### Testing

```bash
# Ejecutar todos los tests
python test_suites/run_tests.py

# Tests con cobertura
python test_suites/run_tests.py --coverage

# Tests específicos
python test_suites/run_tests.py --suite crypto --verbose

# Tests de rendimiento
python test_suites/run_tests.py --suite performance --timeout 600
```

## 📊 Monitoreo y Métricas

### Dashboard Web

El dashboard proporciona:
- **Monitoreo en tiempo real** de CPU, memoria, red
- **Gráficos interactivos** con histórico de métricas
- **Panel de alertas** con clasificación por severidad
- **Visor de logs** con filtrado avanzado
- **Estado de servicios** con indicadores visuales

### API de Métricas

```bash
# Salud general del sistema
GET /api/v1/health

# Métricas detalladas
GET /api/v1/metrics

# Alertas activas
GET /api/v1/alerts

# Estado de servicios
GET /api/v1/services/status
```

### Alertas Inteligentes

El sistema incluye análisis IA para:
- **Detección de patrones** anómalos
- **Alertas predictivas** basadas en tendencias
- **Clasificación automática** de severidad
- **Correlación de eventos** multi-servicio

## 🔒 Seguridad

### Criptografía

- **Algoritmos**: ChaCha20-Poly1305, Blake3, Ed25519
- **Rotación automática** de claves cada 24h
- **Preparación post-cuántica** con algoritmos NIST
- **Verificación de integridad** en todas las comunicaciones

### Red P2P

- **Autenticación mutua** con certificados
- **Cifrado end-to-end** de todos los mensajes
- **Detección de nodos maliciosos** con reputación
- **Soporte TOR** para anonimato opcional

### Consenso

- **Proof of Contribution + PBFT** híbrido
- **Tolerancia a fallas bizantinas** (f < n/3)
- **Verificación criptográfica** de transacciones
- **Prevención de double-spending** y replay attacks

## 🔄 Backup y Recuperación

### Sistema de Backup

- **Cifrado AES-256** de todos los backups
- **Compresión inteligente** (tar.gz, bz2, zip)
- **Sincronización cloud** (AWS S3, Azure, GCP)
- **Rotación automática** con políticas de retención
- **Verificación de integridad** con checksums

### Recuperación

```bash
# Listar backups disponibles
python -c "from backup_system import *; list_backups()"

# Restaurar backup específico
python -c "from backup_system import *; restore_backup('backup_id')"

# Verificar integridad
python -c "from backup_system import *; verify_backup('backup_id')"
```

## 🚨 Solución de Problemas

### Problemas Comunes

**Error: "TOR no disponible"**
```bash
# Ubuntu/Debian
sudo apt install tor
sudo systemctl start tor

# Windows
# Descargar Tor Expert Bundle
# Configurar SOCKS proxy en puerto 9050
```

**Error: "P2P discovery failed"**
```bash
# Verificar firewall
sudo ufw allow 8001/tcp

# Verificar conectividad
python -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', 8001))"
```

**Error: "Crypto module failed"**
```bash
# Instalar dependencias criptográficas
pip install cryptography pynacl

# Verificar permisos de archivos
chmod 600 ~/.aegis/keys/*
```

### Logs y Debugging

```bash
# Logs detallados
LOG_LEVEL=DEBUG python main.py start-node

# Logs específicos de módulo
grep "crypto" logs/aegis.log

# Análisis de rendimiento
python test_suites/run_tests.py --suite performance --profile
```

### Contacto y Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/aegis/issues)
- **Documentación**: [Wiki](https://github.com/tu-usuario/aegis/wiki)
- **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/aegis/discussions)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor lee [CONTRIBUTING.md](CONTRIBUTING.md) para más información sobre cómo contribuir al proyecto.

---

**AEGIS** - Protegiendo la información con tecnología de vanguardia 🛡️