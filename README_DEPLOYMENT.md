# 🛡️ AEGIS Framework - Guía de Despliegue Rápido

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/your-repo/aegis-framework)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#)

> **Sistema de Comunicación Segura y Gestión de Identidad Descentralizada**
> 
> Framework completo para comunicaciones P2P seguras, autenticación blockchain y gestión de identidad descentralizada con integración Tor.

## 🚀 Instalación Rápida

### Windows (PowerShell)
```powershell
# Descargar e instalar automáticamente
git clone https://github.com/your-repo/aegis-framework.git
cd aegis-framework
.\scripts\auto-deploy-windows.ps1
```

### Linux (Bash)
```bash
# Descargar e instalar automáticamente
git clone https://github.com/your-repo/aegis-framework.git
cd aegis-framework
chmod +x scripts/auto-deploy-linux.sh
./scripts/auto-deploy-linux.sh
```

## 📋 Requisitos del Sistema

### Mínimos
- **OS**: Windows 10+ / Ubuntu 18.04+ / CentOS 7+
- **RAM**: 4 GB
- **Disco**: 2 GB libres
- **Red**: Conexión a Internet

### Recomendados
- **OS**: Windows 11 / Ubuntu 22.04+ / CentOS 8+
- **RAM**: 8 GB+
- **Disco**: 5 GB+ libres
- **CPU**: 4 cores+

## 🎯 Inicio Rápido

### 1. Instalar Dependencias
```bash
# El script de auto-despliegue instala automáticamente:
# - Python 3.11+
# - Node.js 20 LTS
# - Git
# - Tor (opcional)
# - Docker (opcional)
```

### 2. Iniciar Servicios

**Windows:**
```powershell
.\scripts\start-all-services.ps1
```

**Linux:**
```bash
./scripts/start-all-services.sh
```

### 3. Acceder a las Aplicaciones

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Dashboard Principal** | http://localhost:8080 | Panel de control AEGIS |
| **Secure Chat UI** | http://localhost:5173 | Interfaz de chat seguro |
| **Blockchain RPC** | http://localhost:8545 | Nodo blockchain local |
| **Tor SOCKS** | 127.0.0.1:9050 | Proxy SOCKS5 |

## 🛠️ Scripts Disponibles

### Despliegue
- `auto-deploy-windows.ps1` - Instalación automática Windows
- `auto-deploy-linux.sh` - Instalación automática Linux

### Gestión de Servicios
- `start-all-services.ps1/.sh` - Iniciar todos los servicios
- `stop-all-services.ps1/.sh` - Detener todos los servicios

### Verificación
- `verify-deployment-windows.ps1` - Verificar instalación Windows
- `verify-deployment-linux.sh` - Verificar instalación Linux

## 📁 Estructura del Proyecto

```
aegis-framework/
├── 📁 config/                 # Configuraciones
│   ├── app_config.json       # Config principal
│   ├── torrc                 # Config Tor
│   └── .env                  # Variables de entorno
├── 📁 dapps/                 # Aplicaciones descentralizadas
│   ├── secure-chat/          # Chat seguro P2P
│   └── aegis-token/          # Token y contratos
├── 📁 scripts/               # Scripts de automatización
├── 📁 logs/                  # Archivos de log
├── 📁 docs/                  # Documentación
└── main.py                   # Aplicación principal
```

## ⚙️ Configuración

### Variables de Entorno (.env)
```bash
# Blockchain
PRIVATE_KEY=your_private_key_here
RPC_URL=http://localhost:8545

# Tor Configuration
TOR_ENABLED=true
TOR_SOCKS_PORT=9050
TOR_CONTROL_PORT=9051

# Security
ENCRYPTION_KEY=your_encryption_key_here
JWT_SECRET=your_jwt_secret_here

# Network
P2P_PORT=8888
API_PORT=8080
```

### Configuración Tor (config/torrc)
```
# Puerto SOCKS
SocksPort 9050

# Puerto de control
ControlPort 9051

# Directorio de datos
DataDirectory ./tor_data

# Configuración de seguridad
CookieAuthentication 1
```

## 🔧 Comandos Útiles

### Verificar Estado
```bash
# Windows
Get-Process python, node, tor -ErrorAction SilentlyContinue

# Linux
ps aux | grep -E '(python.*main.py|node.*(vite|hardhat)|tor.*torrc)'
```

### Ver Logs
```bash
# Logs principales
tail -f logs/dashboard.log
tail -f logs/secure-chat.log
tail -f logs/blockchain.log
tail -f logs/tor.log

# Logs de error
tail -f logs/error.log
```

### Reiniciar Servicios
```bash
# Detener
./scripts/stop-all-services.sh

# Iniciar
./scripts/start-all-services.sh
```

## 🚨 Solución de Problemas

### Problemas Comunes

#### Puerto Ocupado
```bash
# Verificar qué proceso usa el puerto
netstat -tulpn | grep :8080
# o
lsof -i :8080

# Detener proceso específico
kill -9 <PID>
```

#### Error de Dependencias
```bash
# Reinstalar dependencias Python
source venv/bin/activate  # Linux
# o
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

#### Error de Tor
```bash
# Verificar configuración
tor --verify-config -f config/torrc

# Reiniciar Tor
pkill tor
tor -f config/torrc
```

### Logs de Diagnóstico
```bash
# Ejecutar verificación completa
./scripts/verify-deployment-linux.sh --verbose

# Ver reporte JSON
cat logs/deployment_verification.json
```

## 🔒 Seguridad

### Configuración Recomendada
- ✅ Usar Tor para anonimato
- ✅ Generar claves únicas
- ✅ Configurar firewall
- ✅ Actualizar regularmente
- ✅ Monitorear logs

### Puertos de Red
| Puerto | Servicio | Acceso |
|--------|----------|--------|
| 8080 | Dashboard | Local |
| 5173 | Secure Chat | Local |
| 8545 | Blockchain | Local |
| 9050 | Tor SOCKS | Local |
| 9051 | Tor Control | Local |
| 8888 | P2P Network | Externo |

## 📚 Documentación Completa

- 📖 [Guía de Despliegue Completa](DEPLOYMENT_GUIDE_COMPLETE.md)
- 🔧 [Guía de Dependencias](DEPENDENCIES_GUIDE.md)
- 🚨 [Guía de Solución de Problemas](TROUBLESHOOTING_GUIDE.md)
- 🏗️ [Documentación de Arquitectura](docs/ARCHITECTURE.md)
- 🔐 [Guía de Seguridad](docs/SECURITY.md)

## 🤝 Soporte

### Canales de Ayuda
- 📧 **Email**: support@aegis-framework.com
- 💬 **Discord**: [Servidor AEGIS](https://discord.gg/aegis)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/aegis-framework/issues)
- 📖 **Wiki**: [Documentación Wiki](https://github.com/your-repo/aegis-framework/wiki)

### Información del Sistema
```bash
# Generar reporte de diagnóstico
./scripts/verify-deployment-linux.sh --report > system_report.txt

# Incluir en reporte de bug:
# - Versión del OS
# - Logs de error
# - Configuración (sin claves privadas)
# - Pasos para reproducir
```

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📊 Estado del Proyecto

- ✅ **Core Framework**: Completado
- ✅ **Secure Chat**: Completado
- ✅ **Blockchain Integration**: Completado
- ✅ **Tor Integration**: Completado
- 🔄 **Mobile App**: En desarrollo
- 🔄 **Advanced Encryption**: En desarrollo

---

<div align="center">

**🛡️ AEGIS Framework - Comunicación Segura para Todos 🛡️**

[Website](https://aegis-framework.com) • [Documentation](docs/) • [Community](https://discord.gg/aegis)

</div>