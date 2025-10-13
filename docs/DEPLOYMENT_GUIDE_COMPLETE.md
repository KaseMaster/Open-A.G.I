# 🚀 Guía Completa de Despliegue AEGIS Framework

## 📋 Índice
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Despliegue Automático](#despliegue-automático)
- [Despliegue Manual](#despliegue-manual)
- [Configuración Post-Despliegue](#configuración-post-despliegue)
- [Verificación del Sistema](#verificación-del-sistema)
- [Troubleshooting](#troubleshooting)

---

## 🖥️ Requisitos del Sistema

### Windows 10/11
- **Sistema Operativo:** Windows 10 (build 1903+) o Windows 11
- **PowerShell:** 7.0+ (recomendado)
- **Memoria RAM:** Mínimo 8GB, recomendado 16GB
- **Espacio en Disco:** Mínimo 10GB libres
- **Conexión a Internet:** Requerida para descarga de dependencias

### Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+)
- **Kernel:** 5.4+ (recomendado)
- **Memoria RAM:** Mínimo 8GB, recomendado 16GB
- **Espacio en Disco:** Mínimo 10GB libres
- **Conexión a Internet:** Requerida para descarga de dependencias

### Dependencias Principales
- **Python:** 3.8+ (recomendado 3.11)
- **Node.js:** 18+ (recomendado 20 LTS)
- **Git:** 2.30+
- **Docker:** 20.10+ (opcional pero recomendado)
- **Tor:** 0.4.6+ (se instala automáticamente)

---

## 🤖 Despliegue Automático

### Windows
```powershell
# Ejecutar como Administrador
.\scripts\auto-deploy-windows.ps1
```

### Linux
```bash
# Ejecutar con permisos sudo
sudo ./scripts/auto-deploy-linux.sh
```

### Parámetros de Despliegue
```bash
# Despliegue completo (por defecto)
./auto-deploy.sh --mode full

# Solo dependencias
./auto-deploy.sh --mode deps

# Solo configuración
./auto-deploy.sh --mode config

# Desarrollo (incluye herramientas de debug)
./auto-deploy.sh --mode dev

# Producción (optimizado)
./auto-deploy.sh --mode prod
```

---

## 🔧 Despliegue Manual

### 1. Preparación del Entorno

#### Windows
```powershell
# Verificar PowerShell
$PSVersionTable.PSVersion

# Habilitar ejecución de scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Instalar Chocolatey (si no está instalado)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### Linux
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
sudo yum update -y                      # CentOS/RHEL

# Instalar herramientas básicas
sudo apt install -y curl wget git build-essential  # Ubuntu/Debian
sudo yum groupinstall -y "Development Tools"       # CentOS/RHEL
```

### 2. Instalación de Dependencias

#### Python
```bash
# Windows (Chocolatey)
choco install python --version=3.11.0

# Linux
sudo apt install python3.11 python3.11-pip python3.11-venv  # Ubuntu/Debian
sudo yum install python311 python311-pip                     # CentOS/RHEL

# Verificar instalación
python3 --version
pip3 --version
```

#### Node.js
```bash
# Windows (Chocolatey)
choco install nodejs --version=20.10.0

# Linux (usando NodeSource)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs  # Ubuntu/Debian

# Verificar instalación
node --version
npm --version
```

#### Tor
```bash
# Windows (Chocolatey)
choco install tor

# Linux
sudo apt install tor  # Ubuntu/Debian
sudo yum install tor  # CentOS/RHEL (requiere EPEL)

# Verificar instalación
tor --version
```

### 3. Clonación y Configuración del Proyecto

```bash
# Clonar repositorio
git clone https://github.com/KaseMaster/Open-A.G.I.git
cd Open-A.G.I

# Crear entorno virtual Python
python3 -m venv venv

# Activar entorno virtual
# Windows
.\venv\Scripts\Activate.ps1
# Linux
source venv/bin/activate

# Instalar dependencias Python
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Instalar dependencias Node.js para secure-chat
cd dapps/secure-chat/ui
npm install
cd ../../..

# Instalar dependencias Node.js para aegis-token
cd dapps/aegis-token
npm install
cd ../..
```

### 4. Configuración Inicial

```bash
# Copiar archivos de configuración
cp .env.example .env
cp config/config.example.yml config/config.yml

# Generar claves de autenticación Tor
python generate_client_auth.py

# Configurar Tor
sudo mkdir -p /var/lib/tor/aegis_service
sudo chown -R tor:tor /var/lib/tor/aegis_service
```

---

## ⚙️ Configuración Post-Despliegue

### 1. Variables de Entorno

Editar el archivo `.env`:
```bash
# Configuración básica
AEGIS_ENV=production
AEGIS_DEBUG=false
AEGIS_LOG_LEVEL=INFO

# Configuración de red
AEGIS_HOST=0.0.0.0
AEGIS_PORT=8080
AEGIS_TOR_PORT=9050

# Configuración de base de datos
AEGIS_DB_TYPE=sqlite
AEGIS_DB_PATH=./data/aegis.db

# Configuración de seguridad
AEGIS_SECRET_KEY=your-secret-key-here
AEGIS_JWT_SECRET=your-jwt-secret-here

# Configuración de Tor
TOR_CONTROL_PORT=9051
TOR_CONTROL_PASSWORD=your-tor-password
```

### 2. Configuración de Tor

Editar `config/torrc`:
```
# Puerto de control
ControlPort 9051
HashedControlPassword 16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C

# Servicio oculto
HiddenServiceDir /var/lib/tor/aegis_service/
HiddenServicePort 80 127.0.0.1:8080
HiddenServicePort 3000 127.0.0.1:3000

# Configuración de rendimiento
NumEntryGuards 8
CircuitBuildTimeout 30
LearnCircuitBuildTimeout 0
```

### 3. Configuración del Dashboard

Editar `config/app_config.json`:
```json
{
  "dashboard": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8080,
    "debug": false
  },
  "tor": {
    "enabled": true,
    "control_port": 9051,
    "socks_port": 9050
  },
  "p2p": {
    "enabled": true,
    "port": 7777,
    "max_peers": 50
  },
  "security": {
    "encryption": "AES-256-GCM",
    "key_rotation": 3600
  }
}
```

---

## ✅ Verificación del Sistema

### 1. Verificación Automática
```bash
# Ejecutar script de verificación
python scripts/system_validation_test.py

# Verificar servicios específicos
python scripts/test_dashboard_access.py
python scripts/test_onion_access.py
```

### 2. Verificación Manual

#### Servicios Base
```bash
# Verificar Python
python3 --version
pip3 list | grep -E "(flask|requests|cryptography)"

# Verificar Node.js
node --version
npm list -g --depth=0

# Verificar Tor
systemctl status tor  # Linux
Get-Service tor       # Windows
```

#### Servicios AEGIS
```bash
# Iniciar servicios
python main.py start-dashboard --config ./config/app_config.json

# Verificar dashboard (nueva terminal)
curl http://localhost:8080/health

# Verificar secure-chat UI
cd dapps/secure-chat/ui
npm run dev
# Abrir http://localhost:5173

# Verificar blockchain local
cd dapps/aegis-token
npx hardhat node
# Verificar http://localhost:8545
```

### 3. Pruebas de Conectividad

```bash
# Probar conexión Tor
curl --socks5 127.0.0.1:9050 http://check.torproject.org

# Probar servicio oculto
curl --socks5 127.0.0.1:9050 http://your-onion-address.onion

# Probar P2P
python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = s.connect_ex(('127.0.0.1', 7777))
print('P2P OK' if result == 0 else 'P2P Error')
s.close()
"
```

---

## 🚀 Inicio de Servicios

### Modo Desarrollo
```bash
# Terminal 1: Dashboard principal
python main.py start-dashboard --config ./config/app_config.json

# Terminal 2: Tor
tor -f ./config/torrc

# Terminal 3: Secure Chat UI
cd dapps/secure-chat/ui && npm run dev

# Terminal 4: Blockchain local
cd dapps/aegis-token && npx hardhat node
```

### Modo Producción
```bash
# Usar Docker Compose
docker-compose -f docker-compose.yml up -d

# O usar scripts de producción
./scripts/start_production.sh
```

---

## 🔍 Monitoreo y Logs

### Ubicación de Logs
```
logs/
├── aegis_main.log          # Log principal
├── tor_service.log         # Logs de Tor
├── p2p_network.log         # Logs de red P2P
├── dashboard.log           # Logs del dashboard
└── security_events.log     # Eventos de seguridad
```

### Comandos de Monitoreo
```bash
# Logs en tiempo real
tail -f logs/aegis_main.log

# Buscar errores
grep -i error logs/*.log

# Estadísticas del sistema
python scripts/system_stats.py

# Monitoreo de red
netstat -tulpn | grep -E "(8080|9050|9051|7777)"
```

---

## 🛠️ Troubleshooting Común

### Error: Puerto en Uso
```bash
# Encontrar proceso usando el puerto
netstat -tulpn | grep :8080
lsof -i :8080  # Linux/macOS
netstat -ano | findstr :8080  # Windows

# Terminar proceso
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows
```

### Error: Tor No Inicia
```bash
# Verificar configuración
tor --verify-config -f ./config/torrc

# Verificar permisos
sudo chown -R tor:tor /var/lib/tor/
sudo chmod 700 /var/lib/tor/aegis_service/

# Reiniciar servicio
sudo systemctl restart tor
```

### Error: Dependencias Python
```bash
# Reinstalar dependencias
pip install --force-reinstall -r requirements.txt

# Verificar versiones
pip list --outdated

# Limpiar cache
pip cache purge
```

### Error: Node.js/NPM
```bash
# Limpiar cache npm
npm cache clean --force

# Reinstalar node_modules
rm -rf node_modules package-lock.json
npm install

# Verificar versión Node.js
node --version  # Debe ser 18+
```

---

## 📞 Soporte y Contacto

- **Documentación:** [docs/](./docs/)
- **Issues:** [GitHub Issues](https://github.com/KaseMaster/Open-A.G.I/issues)
- **Discusiones:** [GitHub Discussions](https://github.com/KaseMaster/Open-A.G.I/discussions)
- **Security:** Reportar a security@aegis-framework.org

---

## 📄 Licencia

Este proyecto está licenciado bajo MIT License. Ver [LICENSE](../LICENSE) para más detalles.

---

*Última actualización: Diciembre 2024*
*Versión: 2.0.0*