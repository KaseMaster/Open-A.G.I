# 🔧 Guía de Troubleshooting - AEGIS Framework

## 📋 Índice

1. [Problemas Comunes de Instalación](#problemas-comunes-de-instalación)
2. [Errores de Dependencias](#errores-de-dependencias)
3. [Problemas de Configuración](#problemas-de-configuración)
4. [Errores de Red y Puertos](#errores-de-red-y-puertos)
5. [Problemas de Servicios](#problemas-de-servicios)
6. [Errores de Tor](#errores-de-tor)
7. [Problemas de Blockchain](#problemas-de-blockchain)
8. [Errores de UI/Frontend](#errores-de-uifrontend)
9. [Problemas de Permisos](#problemas-de-permisos)
10. [Logs y Diagnóstico](#logs-y-diagnóstico)
11. [Herramientas de Diagnóstico](#herramientas-de-diagnóstico)
12. [Contacto y Soporte](#contacto-y-soporte)

---

## 🚨 Problemas Comunes de Instalación

### Error: "Python no encontrado"

**Síntomas:**
```bash
'python' is not recognized as an internal or external command
python: command not found
```

**Solución Windows:**
```powershell
# Instalar Python usando Chocolatey
choco install python --version=3.11.0

# O descargar desde python.org
# Asegurar que Python esté en PATH
$env:PATH += ";C:\Python311;C:\Python311\Scripts"
```

**Solución Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# CentOS/RHEL/Fedora
sudo dnf install python3.11 python3-pip

# Arch Linux
sudo pacman -S python python-pip
```

### Error: "Node.js versión incompatible"

**Síntomas:**
```bash
Node.js version 16.x detected. Required: 18.x or higher
```

**Solución Windows:**
```powershell
# Desinstalar versión anterior
choco uninstall nodejs

# Instalar versión LTS
choco install nodejs-lts
```

**Solución Linux:**
```bash
# Usar NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# O usar nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
nvm use --lts
```

### Error: "Git no instalado"

**Síntomas:**
```bash
'git' is not recognized as an internal or external command
git: command not found
```

**Solución Windows:**
```powershell
choco install git
```

**Solución Linux:**
```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL/Fedora
sudo dnf install git

# Arch Linux
sudo pacman -S git
```

---

## 📦 Errores de Dependencias

### Error: "pip install failed"

**Síntomas:**
```bash
ERROR: Could not install packages due to an EnvironmentError
Permission denied
```

**Solución:**
```bash
# Usar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias en el entorno virtual
pip install -r requirements.txt

# Si persiste el error, actualizar pip
python -m pip install --upgrade pip
```

### Error: "npm install failed"

**Síntomas:**
```bash
npm ERR! code EACCES
npm ERR! syscall mkdir
npm ERR! path /usr/local/lib/node_modules
```

**Solución:**
```bash
# Configurar npm para usar directorio local
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
export PATH=~/.npm-global/bin:$PATH

# O usar npx para ejecutar paquetes
npx create-react-app my-app
```

### Error: "Tor no instalado"

**Síntomas:**
```bash
'tor' is not recognized as an internal or external command
tor: command not found
```

**Solución Windows:**
```powershell
# Descargar Tor Expert Bundle
# https://www.torproject.org/download/tor/
# Extraer y agregar al PATH

# O usar Chocolatey (no oficial)
choco install tor
```

**Solución Linux:**
```bash
# Ubuntu/Debian
sudo apt install tor

# CentOS/RHEL/Fedora
sudo dnf install tor

# Arch Linux
sudo pacman -S tor
```

---

## ⚙️ Problemas de Configuración

### Error: "Archivo .env no encontrado"

**Síntomas:**
```bash
FileNotFoundError: [Errno 2] No such file or directory: '.env'
```

**Solución:**
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar configuración
nano .env  # Linux
notepad .env  # Windows
```

**Configuración mínima .env:**
```env
# Configuración básica
DEBUG=true
LOG_LEVEL=INFO

# Configuración de red
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8080
UI_HOST=localhost
UI_PORT=5173

# Configuración de Tor
TOR_ENABLED=true
TOR_SOCKS_PORT=9050
TOR_CONTROL_PORT=9051

# Configuración de Blockchain
BLOCKCHAIN_ENABLED=true
BLOCKCHAIN_PORT=8545
```

### Error: "Configuración JSON inválida"

**Síntomas:**
```bash
json.decoder.JSONDecodeError: Expecting ',' delimiter
```

**Solución:**
```bash
# Validar JSON
python -m json.tool config/app_config.json

# O usar jq en Linux
jq . config/app_config.json
```

**Configuración válida app_config.json:**
```json
{
    "dashboard": {
        "host": "localhost",
        "port": 8080,
        "debug": true
    },
    "secure_chat": {
        "host": "localhost",
        "port": 5173,
        "encryption_enabled": true
    },
    "tor": {
        "enabled": true,
        "socks_port": 9050,
        "control_port": 9051,
        "data_directory": "./tor_data"
    },
    "blockchain": {
        "enabled": true,
        "network": "localhost",
        "port": 8545,
        "chain_id": 31337
    }
}
```

---

## 🌐 Errores de Red y Puertos

### Error: "Puerto ya en uso"

**Síntomas:**
```bash
Error: listen EADDRINUSE: address already in use :::8080
OSError: [Errno 98] Address already in use
```

**Solución Windows:**
```powershell
# Encontrar proceso usando el puerto
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# O cambiar puerto en configuración
```

**Solución Linux:**
```bash
# Encontrar proceso usando el puerto
sudo lsof -i :8080
sudo kill -9 <PID>

# O usar fuser
sudo fuser -k 8080/tcp
```

### Error: "Firewall bloqueando conexiones"

**Síntomas:**
```bash
Connection refused
Connection timeout
```

**Solución Windows:**
```powershell
# Agregar regla de firewall
New-NetFirewallRule -DisplayName "AEGIS Dashboard" -Direction Inbound -Port 8080 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "AEGIS SecureChat" -Direction Inbound -Port 5173 -Protocol TCP -Action Allow
```

**Solución Linux:**
```bash
# UFW (Ubuntu)
sudo ufw allow 8080
sudo ufw allow 5173
sudo ufw allow 8545

# iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 5173 -j ACCEPT
```

---

## 🔧 Problemas de Servicios

### Error: "Dashboard no inicia"

**Síntomas:**
```bash
ModuleNotFoundError: No module named 'flask'
ImportError: cannot import name 'Flask'
```

**Solución:**
```bash
# Activar entorno virtual
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows

# Instalar dependencias faltantes
pip install flask flask-cors flask-socketio

# Verificar instalación
python -c "import flask; print(flask.__version__)"
```

### Error: "Secure Chat UI no carga"

**Síntomas:**
```bash
Module not found: Can't resolve 'react'
npm ERR! missing script: dev
```

**Solución:**
```bash
# Navegar al directorio correcto
cd dapps/secure-chat/ui

# Instalar dependencias
npm install

# Verificar package.json
cat package.json | grep -A 5 "scripts"

# Iniciar en modo desarrollo
npm run dev
```

### Error: "Blockchain no conecta"

**Síntomas:**
```bash
Error: could not detect network
Connection refused at 127.0.0.1:8545
```

**Solución:**
```bash
# Navegar al directorio de blockchain
cd dapps/aegis-token

# Instalar dependencias
npm install

# Compilar contratos
npx hardhat compile

# Iniciar red local
npx hardhat node

# En otra terminal, desplegar contratos
npx hardhat run scripts/deploy.js --network localhost
```

---

## 🧅 Errores de Tor

### Error: "Tor no inicia"

**Síntomas:**
```bash
[warn] Failed to bind one of the listener ports.
[err] Couldn't bind to 127.0.0.1:9050: Address already in use
```

**Solución:**
```bash
# Verificar si Tor ya está ejecutándose
ps aux | grep tor  # Linux
tasklist | findstr tor  # Windows

# Detener instancia existente
sudo systemctl stop tor  # Linux (systemd)
sudo service tor stop    # Linux (sysv)

# Usar configuración personalizada
tor -f ./config/torrc
```

### Error: "Tor Control Port no responde"

**Síntomas:**
```bash
stem.SocketError: [Errno 111] Connection refused
```

**Solución:**

**Verificar configuración torrc:**
```bash
# config/torrc debe contener:
ControlPort 9051
CookieAuthentication 1
DataDirectory ./tor_data
```

**Crear directorio de datos:**
```bash
mkdir -p tor_data
chmod 700 tor_data  # Linux
```

### Error: "Tor SOCKS Proxy no funciona"

**Síntomas:**
```bash
SOCKS connection failed
Proxy connection refused
```

**Solución:**
```bash
# Verificar que Tor esté escuchando
netstat -an | grep 9050  # Linux
netstat -an | findstr 9050  # Windows

# Probar conexión SOCKS
curl --socks5 127.0.0.1:9050 http://httpbin.org/ip
```

---

## ⛓️ Problemas de Blockchain

### Error: "Hardhat no encontrado"

**Síntomas:**
```bash
'hardhat' is not recognized as an internal or external command
npx: installed 1 in 2.345s
hardhat: command not found
```

**Solución:**
```bash
# Instalar Hardhat globalmente
npm install -g hardhat

# O usar npx (recomendado)
npx hardhat --version

# Verificar instalación local
cd dapps/aegis-token
npm list hardhat
```

### Error: "Compilación de contratos falla"

**Síntomas:**
```bash
Error HH600: Compilation failed
ParserError: Expected pragma, import directive or contract
```

**Solución:**
```bash
# Verificar versión de Solidity
npx hardhat --version

# Limpiar cache
npx hardhat clean

# Recompilar
npx hardhat compile

# Verificar configuración hardhat.config.js
cat hardhat.config.js
```

### Error: "Red local no disponible"

**Síntomas:**
```bash
Error: could not detect network (event="noNetwork")
ProviderError: Must be authenticated
```

**Solución:**
```bash
# Iniciar red local en terminal separado
npx hardhat node

# Verificar que esté ejecutándose
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545
```

---

## 🎨 Errores de UI/Frontend

### Error: "Vite no inicia"

**Síntomas:**
```bash
Error: Cannot find module 'vite'
ENOENT: no such file or directory, open 'vite.config.js'
```

**Solución:**
```bash
# Navegar al directorio correcto
cd dapps/secure-chat/ui

# Instalar Vite y dependencias
npm install vite @vitejs/plugin-react

# Verificar configuración
cat vite.config.js

# Iniciar servidor de desarrollo
npm run dev
```

### Error: "React componentes no cargan"

**Síntomas:**
```bash
Module not found: Can't resolve 'react'
ReferenceError: React is not defined
```

**Solución:**
```bash
# Instalar React y dependencias
npm install react react-dom

# Verificar versiones
npm list react react-dom

# Limpiar cache de npm
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Error: "CSS/Tailwind no funciona"

**Síntomas:**
```bash
Unknown at rule @tailwind
Tailwind CSS classes not applying
```

**Solución:**
```bash
# Instalar Tailwind CSS
npm install -D tailwindcss postcss autoprefixer

# Generar configuración
npx tailwindcss init -p

# Verificar tailwind.config.js
cat tailwind.config.js

# Verificar CSS principal
cat src/index.css
```

---

## 🔐 Problemas de Permisos

### Error: "Permisos insuficientes (Linux)"

**Síntomas:**
```bash
Permission denied
EACCES: permission denied, mkdir
```

**Solución:**
```bash
# Cambiar propietario del directorio
sudo chown -R $USER:$USER /path/to/aegis

# Cambiar permisos
chmod -R 755 /path/to/aegis

# Para directorios específicos
chmod 700 tor_data
chmod 755 logs
chmod 644 config/*.json
```

### Error: "Permisos de ejecución (Windows)"

**Síntomas:**
```powershell
Execution of scripts is disabled on this system
```

**Solución:**
```powershell
# Cambiar política de ejecución (como administrador)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# O para el script específico
Unblock-File -Path .\auto-deploy-windows.ps1
```

---

## 📊 Logs y Diagnóstico

### Ubicación de Logs

**Estructura de logs:**
```
logs/
├── dashboard.log          # Logs del dashboard principal
├── secure-chat.log        # Logs de la UI de chat seguro
├── blockchain.log         # Logs de la blockchain local
├── tor.log               # Logs de Tor
├── deployment.log        # Logs de despliegue
└── error.log            # Logs de errores generales
```

### Comandos de Diagnóstico

**Ver logs en tiempo real:**
```bash
# Linux
tail -f logs/dashboard.log
tail -f logs/error.log

# Windows
Get-Content logs\dashboard.log -Wait
```

**Buscar errores específicos:**
```bash
# Linux
grep -i error logs/*.log
grep -i "connection refused" logs/*.log

# Windows
Select-String -Pattern "error" -Path logs\*.log
```

### Niveles de Log

**Configurar nivel de detalle en .env:**
```env
LOG_LEVEL=DEBUG    # Máximo detalle
LOG_LEVEL=INFO     # Información general
LOG_LEVEL=WARNING  # Solo advertencias y errores
LOG_LEVEL=ERROR    # Solo errores críticos
```

---

## 🛠️ Herramientas de Diagnóstico

### Scripts de Verificación

**Ejecutar verificación completa:**
```bash
# Windows
.\scripts\verify-deployment-windows.ps1 --detailed

# Linux
./scripts/verify-deployment-linux.sh --detailed
```

**Verificación específica:**
```bash
# Solo dependencias
.\scripts\verify-deployment-windows.ps1 --skip-services --skip-network

# Solo servicios
.\scripts\verify-deployment-linux.sh --skip-network
```

### Comandos de Red

**Verificar puertos:**
```bash
# Windows
netstat -an | findstr "8080 5173 8545 9050 9051"

# Linux
netstat -tlnp | grep -E "(8080|5173|8545|9050|9051)"
ss -tlnp | grep -E "(8080|5173|8545|9050|9051)"
```

**Probar conectividad:**
```bash
# Probar endpoints HTTP
curl -I http://localhost:8080
curl -I http://localhost:5173

# Probar SOCKS proxy
curl --socks5 127.0.0.1:9050 http://httpbin.org/ip
```

### Herramientas de Sistema

**Monitoreo de recursos:**
```bash
# Linux
htop
iotop
nethogs

# Windows
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10
Get-Counter "\Processor(_Total)\% Processor Time"
```

---

## 🔍 Diagnóstico Avanzado

### Problemas de Memoria

**Síntomas:**
```bash
MemoryError: Unable to allocate array
Out of memory
```

**Solución:**
```bash
# Verificar uso de memoria
free -h  # Linux
Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory  # Windows

# Optimizar configuración
# Reducir workers en config/app_config.json
{
    "dashboard": {
        "workers": 2,
        "max_connections": 100
    }
}
```

### Problemas de Disco

**Síntomas:**
```bash
No space left on device
Disk full
```

**Solución:**
```bash
# Verificar espacio
df -h  # Linux
Get-WmiObject -Class Win32_LogicalDisk  # Windows

# Limpiar logs antiguos
find logs/ -name "*.log" -mtime +7 -delete  # Linux
Get-ChildItem logs\*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item  # Windows

# Limpiar cache
rm -rf node_modules/.cache  # Linux
rm -rf __pycache__
```

### Problemas de Red Avanzados

**Verificar DNS:**
```bash
# Verificar resolución DNS
nslookup localhost
dig localhost  # Linux

# Probar con IP directa
curl http://127.0.0.1:8080
```

**Verificar routing:**
```bash
# Linux
ip route show
netstat -rn

# Windows
route print
```

---

## 📞 Contacto y Soporte

### Información de Soporte

**Antes de contactar soporte:**
1. Ejecutar script de verificación con `--detailed`
2. Recopilar logs relevantes de `logs/`
3. Documentar pasos para reproducir el problema
4. Incluir información del sistema (OS, versiones, etc.)

### Reportar Bugs

**Información requerida:**
- Versión de AEGIS Framework
- Sistema operativo y versión
- Versiones de Python, Node.js, etc.
- Logs de error completos
- Pasos para reproducir

### Recursos Adicionales

**Documentación:**
- `README.md` - Información general del proyecto
- `DEPLOYMENT_GUIDE_COMPLETE.md` - Guía de despliegue
- `DEPENDENCIES_GUIDE.md` - Guía de dependencias
- **Documentación**: [https://github.com/KaseMaster/Open-A.G.I/wiki](https://github.com/KaseMaster/Open-A.G.I/wiki)

**Scripts útiles:**
- `auto-deploy-windows.ps1` - Instalación automática Windows
- `auto-deploy-linux.sh` - Instalación automática Linux
- `verify-deployment-*.{ps1,sh}` - Scripts de verificación

**Repositorio**: [https://github.com/KaseMaster/Open-A.G.I](https://github.com/KaseMaster/Open-A.G.I)
**Issues**: [https://github.com/KaseMaster/Open-A.G.I/issues](https://github.com/KaseMaster/Open-A.G.I/issues)

---

## 🔄 Procedimientos de Recuperación

### Reinstalación Completa

**Si todo falla, reinstalación limpia:**

```bash
# 1. Hacer backup de configuración
cp -r config config_backup
cp .env .env.backup

# 2. Limpiar instalación
rm -rf venv node_modules dapps/*/node_modules
rm -rf logs tor_data

# 3. Ejecutar instalación automática
# Windows: .\scripts\auto-deploy-windows.ps1
# Linux: ./scripts/auto-deploy-linux.sh

# 4. Restaurar configuración personalizada
cp config_backup/* config/
cp .env.backup .env
```

### Recuperación de Servicios

**Reiniciar todos los servicios:**

```bash
# Detener todos los procesos
pkill -f "python.*main.py"  # Linux
pkill -f "npm run dev"
pkill -f "npx hardhat node"
pkill -f "tor"

# Windows equivalente
taskkill /F /IM python.exe
taskkill /F /IM node.exe
taskkill /F /IM tor.exe

# Reiniciar servicios
./scripts/start-all-services.sh  # Linux
.\scripts\start-all-services.ps1  # Windows
```

---

**💡 Tip:** Mantén este documento actualizado con nuevos problemas y soluciones que encuentres durante el uso del sistema AEGIS.

**🔒 Recuerda:** Siempre verifica que los servicios de seguridad (Tor, encriptación) estén funcionando correctamente antes de usar el sistema en producción.