# 📦 Guía de Configuración de Dependencias AEGIS Framework

## 📋 Índice
- [Dependencias Principales](#dependencias-principales)
- [Configuración por Sistema Operativo](#configuración-por-sistema-operativo)
- [Entornos Virtuales](#entornos-virtuales)
- [Gestión de Versiones](#gestión-de-versiones)
- [Dependencias Opcionales](#dependencias-opcionales)
- [Troubleshooting de Dependencias](#troubleshooting-de-dependencias)

---

## 🔧 Dependencias Principales

### Python 3.8+ (Recomendado 3.11)
```bash
# Verificar versión
python3 --version

# Versiones soportadas
Python 3.8.x  ✅ Mínimo soportado
Python 3.9.x  ✅ Soportado
Python 3.10.x ✅ Soportado
Python 3.11.x ✅ Recomendado
Python 3.12.x ✅ Soportado (experimental)
```

### Node.js 18+ (Recomendado 20 LTS)
```bash
# Verificar versión
node --version

# Versiones soportadas
Node.js 16.x  ⚠️  Deprecado
Node.js 18.x  ✅ Mínimo soportado
Node.js 20.x  ✅ Recomendado (LTS)
Node.js 21.x  ✅ Soportado
```

### Git 2.30+
```bash
# Verificar versión
git --version

# Funcionalidades requeridas
- Git LFS support
- SSH key authentication
- HTTPS authentication
```

### Tor 0.4.6+
```bash
# Verificar versión
tor --version

# Características requeridas
- Hidden services v3
- Control port support
- SOCKS5 proxy
```

---

## 🖥️ Configuración por Sistema Operativo

### Windows 10/11

#### Método 1: Chocolatey (Recomendado)
```powershell
# Instalar Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Instalar dependencias
choco install python --version=3.11.0
choco install nodejs --version=20.10.0
choco install git
choco install tor
```

#### Método 2: Instaladores Oficiales
```powershell
# Python desde python.org
# Descargar: https://www.python.org/downloads/windows/
# Asegurar: "Add Python to PATH" marcado

# Node.js desde nodejs.org
# Descargar: https://nodejs.org/en/download/
# Incluye NPM automáticamente

# Git desde git-scm.com
# Descargar: https://git-scm.com/download/win
# Configurar: Git Bash, Git GUI

# Tor Browser Bundle
# Descargar: https://www.torproject.org/download/
```

#### Configuración de PATH Windows
```powershell
# Verificar PATH actual
$env:PATH -split ';'

# Agregar Python al PATH (si no está)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Python311\;C:\Python311\Scripts\", "User")

# Agregar Node.js al PATH (si no está)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\nodejs\", "User")

# Recargar PATH
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
```

### Ubuntu/Debian

#### Actualización del Sistema
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential software-properties-common
```

#### Python 3.11
```bash
# Agregar repositorio deadsnakes
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Instalar Python 3.11
sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev

# Crear enlaces simbólicos
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1
```

#### Node.js 20 LTS
```bash
# Usar NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verificar instalación
node --version
npm --version
```

#### Tor
```bash
# Agregar repositorio oficial de Tor
sudo apt install -y apt-transport-https
curl -s https://deb.torproject.org/torproject.org/A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89.asc | sudo apt-key add -
echo "deb https://deb.torproject.org/torproject.org $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tor.list

# Instalar Tor
sudo apt update
sudo apt install -y tor deb.torproject.org-keyring

# Habilitar servicio
sudo systemctl enable tor
sudo systemctl start tor
```

### CentOS/RHEL/Fedora

#### Actualización del Sistema
```bash
# CentOS/RHEL 8+
sudo dnf update -y
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y curl wget git

# CentOS/RHEL 7
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y curl wget git
```

#### Python 3.11 (Compilación desde fuente)
```bash
# Instalar dependencias de compilación
sudo dnf install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel

# Descargar y compilar Python 3.11
cd /tmp
wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
tar xzf Python-3.11.7.tgz
cd Python-3.11.7
./configure --enable-optimizations
make altinstall

# Clonar repositorio AEGIS
git clone https://github.com/KaseMaster/Open-A.G.I.git

# Crear enlaces simbólicos
sudo ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3
sudo ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3
```

#### Node.js 20 LTS
```bash
# Usar NodeSource repository
curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
sudo dnf install -y nodejs  # o yum install -y nodejs
```

#### Tor
```bash
# Habilitar EPEL repository
sudo dnf install -y epel-release  # o yum install -y epel-release

# Instalar Tor
sudo dnf install -y tor  # o yum install -y tor

# Habilitar servicio
sudo systemctl enable tor
sudo systemctl start tor
```

---

## 🐍 Entornos Virtuales

### Python Virtual Environment

#### Creación y Activación
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# Linux/macOS
source venv/bin/activate

# Windows
.\venv\Scripts\Activate.ps1
# o
.\venv\Scripts\activate.bat
```

#### Gestión de Dependencias
```bash
# Instalar dependencias del proyecto
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Actualizar pip
pip install --upgrade pip

# Listar paquetes instalados
pip list

# Generar requirements.txt actualizado
pip freeze > requirements-current.txt
```

#### Desactivación
```bash
# Desactivar entorno virtual
deactivate
```

### Node.js Package Management

#### NPM Configuration
```bash
# Verificar configuración NPM
npm config list

# Configurar registry (si es necesario)
npm config set registry https://registry.npmjs.org/

# Limpiar cache
npm cache clean --force

# Verificar cache
npm cache verify
```

#### Instalación de Dependencias
```bash
# Secure Chat UI
cd dapps/secure-chat/ui
npm install
npm audit fix  # Corregir vulnerabilidades

# AEGIS Token
cd dapps/aegis-token
npm install
npm audit fix
```

---

## 📊 Gestión de Versiones

### Python Dependencies (requirements.txt)
```txt
# Core Framework
flask==3.0.0
requests==2.31.0
cryptography==41.0.8
pycryptodome==3.19.0

# Networking & Tor
stem==1.8.2
pysocks==1.7.1
aiohttp==3.9.1

# Database & Storage
sqlalchemy==2.0.23
alembic==1.13.1

# Security & Authentication
pyjwt==2.8.0
bcrypt==4.1.2
passlib==1.7.4

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0

# Monitoring & Logging
loguru==0.7.2
prometheus-client==0.19.0
```

### Node.js Dependencies (package.json)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "ethers": "^6.8.0",
    "hardhat": "^2.19.0",
    "socket.io-client": "^4.7.0"
  },
  "devDependencies": {
    "eslint": "^8.55.0",
    "prettier": "^3.1.0",
    "@types/react": "^18.2.0",
    "typescript": "^5.3.0"
  }
}
```

### Version Pinning Strategy
```bash
# Python - Pin major.minor, allow patch updates
flask>=3.0.0,<3.1.0
requests>=2.31.0,<2.32.0

# Node.js - Use caret for compatible updates
"react": "^18.2.0"     # Allows 18.x.x
"vite": "^5.0.0"       # Allows 5.x.x
```

---

## 🔧 Dependencias Opcionales

### Docker & Docker Compose
```bash
# Instalación Docker (Linux)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Windows (Docker Desktop)
# Descargar desde: https://www.docker.com/products/docker-desktop
```

### Hardhat Development Tools
```bash
cd dapps/aegis-token

# Instalar Hardhat globalmente (opcional)
npm install -g hardhat

# Herramientas adicionales
npm install -g @nomiclabs/hardhat-ethers
npm install -g @nomiclabs/hardhat-waffle
```

### Development Tools
```bash
# Python development tools
pip install black flake8 mypy pytest pytest-cov

# Node.js development tools
npm install -g eslint prettier typescript ts-node

# Git hooks (opcional)
pip install pre-commit
pre-commit install
```

---

## 🛠️ Troubleshooting de Dependencias

### Problemas Comunes de Python

#### Error: "pip not found"
```bash
# Linux/macOS
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

# Windows
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

#### Error: "Permission denied"
```bash
# Usar --user flag
pip install --user package_name

# O usar entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows
```

#### Error: "SSL Certificate verify failed"
```bash
# Temporal (no recomendado para producción)
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org package_name

# Mejor: actualizar certificados
# macOS
/Applications/Python\ 3.11/Install\ Certificates.command

# Linux
sudo apt update && sudo apt install ca-certificates
```

### Problemas Comunes de Node.js

#### Error: "EACCES permissions"
```bash
# Cambiar directorio global de npm
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Error: "Module not found"
```bash
# Limpiar cache y reinstalar
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### Error: "Unsupported engine"
```bash
# Verificar versión de Node.js
node --version

# Actualizar Node.js si es necesario
# Linux (usando n)
sudo npm install -g n
sudo n stable

# Windows (descargar nueva versión)
# https://nodejs.org/en/download/
```

### Problemas de Tor

#### Error: "Tor not starting"
```bash
# Verificar configuración
tor --verify-config -f ./config/torrc

# Verificar permisos
sudo chown -R tor:tor /var/lib/tor/
sudo chmod 700 /var/lib/tor/aegis_service/

# Verificar logs
sudo journalctl -u tor -f
```

#### Error: "Control port not accessible"
```bash
# Verificar puerto en uso
netstat -tulpn | grep 9051

# Generar nueva contraseña
tor --hash-password "your_password"

# Actualizar torrc con nueva contraseña hash
```

### Problemas de Compilación

#### Error: "Microsoft Visual C++ 14.0 is required" (Windows)
```powershell
# Instalar Build Tools para Visual Studio
choco install visualstudio2022buildtools

# O descargar desde Microsoft
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Error: "gcc not found" (Linux)
```bash
# Ubuntu/Debian
sudo apt install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo dnf groupinstall "Development Tools"
```

---

## 📋 Checklist de Verificación

### Pre-instalación
- [ ] Sistema operativo compatible
- [ ] Permisos de administrador/sudo
- [ ] Conexión a internet estable
- [ ] Espacio en disco suficiente (10GB+)
- [ ] RAM suficiente (8GB+)

### Post-instalación Python
- [ ] `python3 --version` muestra 3.8+
- [ ] `pip3 --version` funciona
- [ ] Entorno virtual creado y activado
- [ ] `pip install -r requirements.txt` exitoso
- [ ] Importaciones críticas funcionan

### Post-instalación Node.js
- [ ] `node --version` muestra 18+
- [ ] `npm --version` funciona
- [ ] `npm install` en secure-chat/ui exitoso
- [ ] `npm install` en aegis-token exitoso
- [ ] `npm run dev` inicia sin errores

### Post-instalación Tor
- [ ] `tor --version` muestra 0.4.6+
- [ ] Servicio Tor iniciado
- [ ] Puerto de control accesible
- [ ] Directorio de servicio oculto creado

### Verificación Final
- [ ] Todos los servicios inician correctamente
- [ ] Dashboard accesible en localhost:8080
- [ ] Secure Chat accesible en localhost:5173
- [ ] Blockchain local accesible en localhost:8545
- [ ] Logs no muestran errores críticos

---

## 📞 Soporte Adicional

### Recursos de Documentación
- [Python.org Documentation](https://docs.python.org/3/)
- [Node.js Documentation](https://nodejs.org/en/docs/)
- [Tor Project Documentation](https://www.torproject.org/docs/)
- [Git Documentation](https://git-scm.com/doc)

### Comunidad y Soporte
- **GitHub Issues:** [Reportar problemas específicos](https://github.com/KaseMaster/Open-A.G.I/issues)
- **Discussions:** [Preguntas generales](https://github.com/KaseMaster/Open-A.G.I/discussions)
- **Stack Overflow:** Usar tags `aegis-framework`, `python`, `nodejs`, `tor`

### Logs y Debugging
```bash
# Habilitar logging detallado
export AEGIS_DEBUG=true
export AEGIS_LOG_LEVEL=DEBUG

# Ubicación de logs
tail -f logs/aegis_main.log
tail -f logs/dependency_install.log
```

---

*Última actualización: Diciembre 2024*
*Versión: 2.0.0*