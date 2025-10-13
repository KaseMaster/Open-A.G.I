# 🚀 Guía Completa de Despliegue - AEGIS Framework

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación Rápida](#instalación-rápida)
4. [Despliegue en Desarrollo](#despliegue-en-desarrollo)
5. [Despliegue en Producción](#despliegue-en-producción)
6. [Configuración Avanzada](#configuración-avanzada)
7. [Monitoreo y Mantenimiento](#monitoreo-y-mantenimiento)
8. [Solución de Problemas](#solución-de-problemas)
9. [Seguridad](#seguridad)
10. [Backup y Recuperación](#backup-y-recuperación)

---

## 🎯 Introducción

AEGIS Framework es un sistema de gestión de conocimiento y comunicación segura que incluye:

- **Dashboard Principal**: Interfaz de administración y control
- **Secure Chat UI**: Sistema de chat cifrado
- **Blockchain**: Red blockchain local para tokens AEGIS
- **Tor Integration**: Comunicación anónima y segura

Esta guía cubre todos los aspectos del despliegue, desde desarrollo hasta producción.

---

## 💻 Requisitos del Sistema

### Requisitos Mínimos

#### Windows
- **SO**: Windows 10 Pro/Enterprise o Windows Server 2019+
- **RAM**: 4GB disponible (8GB recomendado)
- **Disco**: 20GB espacio libre (50GB recomendado)
- **CPU**: 2 cores (4 cores recomendado)
- **Red**: Conexión a Internet estable

#### Linux
- **SO**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **RAM**: 4GB disponible (8GB recomendado)
- **Disco**: 20GB espacio libre (50GB recomendado)
- **CPU**: 2 cores (4 cores recomendado)
- **Red**: Conexión a Internet estable

### Puertos Requeridos

| Servicio | Puerto | Protocolo | Descripción |
|----------|--------|-----------|-------------|
| Dashboard | 8080 | HTTP/HTTPS | Interfaz principal |
| Secure Chat UI | 3000 | HTTP/HTTPS | Chat seguro |
| Blockchain RPC | 8545 | HTTP | API blockchain |
| Tor SOCKS | 9050 | SOCKS5 | Proxy Tor |
| Tor Control | 9051 | TCP | Control Tor |

### Dependencias

- **Python**: 3.8+ (3.11 recomendado)
- **Node.js**: 18+ (20 LTS recomendado)
- **npm**: 8+
- **Git**: 2.0+
- **Tor**: Latest stable
- **Docker**: 20+ (opcional)

---

## ⚡ Instalación Rápida

### Windows

```powershell
# 1. Clonar repositorio
git clone https://github.com/your-org/aegis-framework.git
cd aegis-framework

# 2. Instalar dependencias
.\scripts\install-dependencies.ps1

# 3. Configurar sistema
.\scripts\setup-config.ps1

# 4. Verificar instalación
.\scripts\health-check.ps1

# 5. Iniciar servicios
.\scripts\start-all-services.ps1
```

### Linux

```bash
# 1. Clonar repositorio
git clone https://github.com/your-org/aegis-framework.git
cd aegis-framework

# 2. Hacer scripts ejecutables
chmod +x scripts/*.sh

# 3. Instalar dependencias
./scripts/install-dependencies.sh

# 4. Configurar sistema
./scripts/setup-config.sh

# 5. Verificar instalación
./scripts/health-check.sh

# 6. Iniciar servicios
./scripts/start-all-services.sh
```

### Verificación de Instalación

Después de la instalación, verifica que todos los servicios estén funcionando:

```bash
# Verificar estado de servicios
curl http://localhost:8080/health    # Dashboard
curl http://localhost:3000           # Secure Chat UI
curl http://localhost:8545           # Blockchain (si está configurado)
```

---

## 🛠️ Despliegue en Desarrollo

### Configuración de Desarrollo

1. **Crear entorno virtual Python**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate     # Windows
   ```

2. **Instalar dependencias Python**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Dependencias de desarrollo
   ```

3. **Configurar variables de entorno**:
   ```bash
   cp .env.example .env
   # Editar .env con configuraciones de desarrollo
   ```

4. **Instalar dependencias Node.js**:
   ```bash
   # Secure Chat UI
   cd dapps/secure-chat/ui
   npm install
   npm run dev &
   cd ../../..

   # AEGIS Token (opcional)
   cd dapps/aegis-token
   npm install
   cd ../..
   ```

### Iniciar en Modo Desarrollo

```bash
# Opción 1: Usar scripts automatizados
./scripts/start-all-services.sh --dev

# Opción 2: Iniciar servicios individualmente
python main.py start-dashboard --config config/app_config.json --debug
cd dapps/secure-chat/ui && npm run dev
cd dapps/aegis-token && npx hardhat node  # Si usas blockchain
tor -f config/torrc
```

### Hot Reload y Desarrollo

- **Dashboard**: Cambios en Python se recargan automáticamente con `--debug`
- **Secure Chat UI**: Vite proporciona hot reload automático
- **Configuración**: Cambios en `.env` requieren reinicio de servicios

---

## 🏭 Despliegue en Producción

### Preparación para Producción

1. **Verificar requisitos**:
   ```bash
   # Windows
   .\scripts\deploy-production.ps1 --dry-run

   # Linux
   ./scripts/deploy-production.sh --dry-run
   ```

2. **Configurar dominio y SSL** (recomendado):
   ```bash
   # Con dominio y certificados SSL
   ./scripts/deploy-production.sh \
     --domain aegis.company.com \
     --ssl-cert /path/to/cert.pem \
     --ssl-key /path/to/key.pem
   ```

### Despliegue Automático

#### Windows
```powershell
# Despliegue completo
.\scripts\deploy-production.ps1 -Domain "aegis.company.com"

# Despliegue rápido (omitir pruebas y backup)
.\scripts\deploy-production.ps1 -Force -SkipTests -SkipBackup

# Solo verificar sin cambios
.\scripts\deploy-production.ps1 -DryRun
```

#### Linux
```bash
# Despliegue completo
./scripts/deploy-production.sh --domain aegis.company.com

# Despliegue rápido
./scripts/deploy-production.sh --force --skip-tests --skip-backup

# Solo verificar sin cambios
./scripts/deploy-production.sh --dry-run
```

### Configuración Manual de Producción

Si prefieres configurar manualmente:

1. **Variables de entorno de producción**:
   ```bash
   # .env
   FLASK_ENV=production
   DEBUG=False
   SECRET_KEY=your-super-secret-key-here
   DATABASE_URL=postgresql://user:pass@localhost/aegis_prod
   LOG_LEVEL=INFO
   ```

2. **Configuración de aplicación**:
   ```json
   {
     "environment": "production",
     "debug": false,
     "logging": {
       "level": "INFO",
       "file": "logs/aegis-production.log"
     },
     "security": {
       "csrf_enabled": true,
       "session_timeout": 3600
     }
   }
   ```

3. **Build de aplicaciones frontend**:
   ```bash
   cd dapps/secure-chat/ui
   npm ci --production
   npm run build
   ```

### Servicios del Sistema (Linux)

Para configurar AEGIS como servicio del sistema:

```bash
# Crear servicio systemd
sudo tee /etc/systemd/system/aegis-dashboard.service > /dev/null << EOF
[Unit]
Description=AEGIS Framework Dashboard
After=network.target

[Service]
Type=simple
User=aegis
WorkingDirectory=/opt/aegis-framework
Environment=PATH=/opt/aegis-framework/venv/bin
ExecStart=/opt/aegis-framework/venv/bin/python main.py start-dashboard
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Habilitar y iniciar servicio
sudo systemctl daemon-reload
sudo systemctl enable aegis-dashboard.service
sudo systemctl start aegis-dashboard.service
```

---

## ⚙️ Configuración Avanzada

### Configuración de Base de Datos

#### SQLite (Por defecto)
```bash
DATABASE_URL=sqlite:///aegis.db
```

#### PostgreSQL (Recomendado para producción)
```bash
# Instalar PostgreSQL
sudo apt install postgresql postgresql-contrib  # Ubuntu/Debian
# o
brew install postgresql  # macOS

# Crear base de datos
sudo -u postgres createdb aegis_production
sudo -u postgres createuser aegis_user

# Configurar .env
DATABASE_URL=postgresql://aegis_user:password@localhost/aegis_production
```

#### MySQL/MariaDB
```bash
DATABASE_URL=mysql://user:password@localhost/aegis_production
```

### Configuración de Proxy Reverso

#### Nginx
```nginx
server {
    listen 80;
    server_name aegis.company.com;
    
    # Redirigir HTTP a HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aegis.company.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Dashboard
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Secure Chat UI
    location /chat {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### Apache
```apache
<VirtualHost *:80>
    ServerName aegis.company.com
    Redirect permanent / https://aegis.company.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName aegis.company.com
    
    SSLEngine on
    SSLCertificateFile /path/to/cert.pem
    SSLCertificateKeyFile /path/to/key.pem
    
    # Dashboard
    ProxyPass / http://localhost:8080/
    ProxyPassReverse / http://localhost:8080/
    
    # Secure Chat UI
    ProxyPass /chat http://localhost:3000/
    ProxyPassReverse /chat http://localhost:3000/
</VirtualHost>
```

### Configuración de Firewall

#### UFW (Ubuntu)
```bash
# Permitir puertos necesarios
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8080/tcp    # Dashboard (si no usas proxy)
sudo ufw allow 3000/tcp    # Secure Chat (si no usas proxy)

# Habilitar firewall
sudo ufw enable
```

#### iptables
```bash
# Permitir tráfico entrante en puertos específicos
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp --dport 3000 -j ACCEPT

# Guardar reglas
iptables-save > /etc/iptables/rules.v4
```

---

## 📊 Monitoreo y Mantenimiento

### Monitoreo Continuo

```bash
# Monitoreo en tiempo real
./scripts/monitor-services.sh --continuous --interval 30

# Monitoreo con logging
./scripts/monitor-services.sh --continuous --log-file logs/monitoring.log

# Verificación de salud
./scripts/health-check.sh --detailed
```

### Logs y Auditoría

#### Ubicaciones de Logs
- **Dashboard**: `logs/aegis-dashboard.log`
- **Secure Chat**: `dapps/secure-chat/ui/logs/`
- **Blockchain**: `dapps/aegis-token/logs/`
- **Tor**: `logs/tor.log`
- **Sistema**: `logs/system.log`

#### Comandos Útiles
```bash
# Ver logs en tiempo real
tail -f logs/aegis-dashboard.log

# Buscar errores
grep -i error logs/*.log

# Logs de las últimas 24 horas
find logs -name "*.log" -mtime -1 -exec grep -l "ERROR\|WARN" {} \;

# Rotar logs
./scripts/rotate-logs.sh
```

### Métricas del Sistema

```bash
# Uso de recursos
./scripts/health-check.sh --system-metrics

# Estadísticas de red
ss -tuln | grep -E ':(8080|3000|8545|9050)'

# Procesos AEGIS
ps aux | grep -E '(python.*main.py|node.*aegis|tor)'
```

### Actualizaciones

```bash
# Verificar actualizaciones disponibles
./scripts/update-system.sh --check-only

# Actualizar dependencias Python
./scripts/update-system.sh --python

# Actualizar dependencias Node.js
./scripts/update-system.sh --nodejs

# Actualización completa
./scripts/update-system.sh --all --backup
```

---

## 🔧 Solución de Problemas

### Problemas Comunes

#### 1. Servicios no inician

**Síntomas**: Error al iniciar servicios
```bash
# Verificar logs
tail -f logs/aegis-dashboard.log

# Verificar puertos
ss -tuln | grep -E ':(8080|3000|8545)'

# Verificar dependencias
./scripts/health-check.sh --dependencies
```

**Soluciones**:
- Verificar que los puertos no estén en uso
- Comprobar permisos de archivos
- Verificar configuración en `.env`

#### 2. Error de conexión a base de datos

**Síntomas**: `Database connection failed`
```bash
# Verificar configuración
grep DATABASE_URL .env

# Probar conexión
python -c "from sqlalchemy import create_engine; engine = create_engine('your_db_url'); print(engine.execute('SELECT 1').scalar())"
```

**Soluciones**:
- Verificar credenciales de base de datos
- Comprobar que el servicio de BD esté ejecutándose
- Verificar permisos de usuario de BD

#### 3. Problemas con Tor

**Síntomas**: Tor no se conecta o falla
```bash
# Verificar configuración Tor
cat config/torrc

# Verificar logs Tor
tail -f logs/tor.log

# Probar conexión manual
curl --socks5 localhost:9050 http://check.torproject.org/
```

**Soluciones**:
- Verificar configuración de `torrc`
- Comprobar permisos del directorio `tor_data`
- Reiniciar servicio Tor

#### 4. Frontend no carga

**Síntomas**: Página en blanco o errores 404
```bash
# Verificar build
ls -la dapps/secure-chat/ui/dist/

# Verificar logs del servidor
tail -f logs/aegis-dashboard.log

# Probar en modo desarrollo
cd dapps/secure-chat/ui && npm run dev
```

**Soluciones**:
- Ejecutar `npm run build` en el directorio UI
- Verificar configuración de proxy
- Comprobar permisos de archivos estáticos

### Herramientas de Diagnóstico

#### Script de Diagnóstico Completo
```bash
# Crear script de diagnóstico
cat > scripts/diagnose.sh << 'EOF'
#!/bin/bash
echo "=== AEGIS Framework Diagnostic ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo ""

echo "=== System Info ==="
uname -a
free -h
df -h
echo ""

echo "=== Network ==="
ss -tuln | grep -E ':(8080|3000|8545|9050)'
echo ""

echo "=== Processes ==="
ps aux | grep -E '(python.*main.py|node.*aegis|tor)' | grep -v grep
echo ""

echo "=== Recent Errors ==="
find logs -name "*.log" -mtime -1 -exec grep -l "ERROR" {} \; | head -5
echo ""

echo "=== Disk Usage ==="
du -sh logs/ data/ venv/ dapps/
EOF

chmod +x scripts/diagnose.sh
./scripts/diagnose.sh
```

### Recuperación de Desastres

#### Restaurar desde Backup
```bash
# Listar backups disponibles
./scripts/backup-config.sh --list

# Restaurar backup específico
./scripts/backup-config.sh --restore backup-20240101-120000

# Verificar restauración
./scripts/health-check.sh
```

#### Reinstalación Limpia
```bash
# Detener todos los servicios
./scripts/stop-all-services.sh --force

# Backup de datos importantes
./scripts/backup-config.sh

# Limpiar instalación
rm -rf venv/ node_modules/ dapps/*/node_modules/

# Reinstalar
./scripts/install-dependencies.sh
./scripts/setup-config.sh
```

---

## 🔒 Seguridad

### Configuración de Seguridad Básica

#### 1. Cambiar Credenciales por Defecto
```bash
# Generar nueva clave secreta
python -c "import secrets; print(secrets.token_hex(32))"

# Actualizar .env
SECRET_KEY=nueva_clave_generada
SECURITY_PASSWORD_SALT=nueva_salt_generada
```

#### 2. Configurar HTTPS
```bash
# Generar certificado auto-firmado (desarrollo)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Usar Let's Encrypt (producción)
sudo certbot --nginx -d aegis.company.com
```

#### 3. Configurar Firewall
```bash
# Solo permitir puertos necesarios
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Hardening del Sistema

#### 1. Configuración de Usuario
```bash
# Crear usuario dedicado
sudo useradd -m -s /bin/bash aegis
sudo usermod -aG sudo aegis

# Configurar permisos
sudo chown -R aegis:aegis /opt/aegis-framework
sudo chmod 750 /opt/aegis-framework
```

#### 2. Configuración de Logs de Auditoría
```bash
# Habilitar logging detallado
echo "LOG_LEVEL=DEBUG" >> .env
echo "AUDIT_LOGGING=True" >> .env

# Configurar rotación de logs
sudo tee /etc/logrotate.d/aegis << EOF
/opt/aegis-framework/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
```

#### 3. Configuración de Tor Segura
```bash
# Configurar torrc con opciones de seguridad
cat >> config/torrc << EOF
# Seguridad adicional
DisableDebuggerAttachment 1
SafeLogging 1
MaxCircuitDirtiness 600
NewCircuitPeriod 15
MaxClientCircuitsPending 32
EOF
```

### Monitoreo de Seguridad

#### 1. Detección de Intrusiones
```bash
# Instalar fail2ban
sudo apt install fail2ban

# Configurar para AEGIS
sudo tee /etc/fail2ban/jail.local << EOF
[aegis-dashboard]
enabled = true
port = 8080
filter = aegis-dashboard
logpath = /opt/aegis-framework/logs/aegis-dashboard.log
maxretry = 5
bantime = 3600
EOF
```

#### 2. Monitoreo de Archivos
```bash
# Instalar AIDE
sudo apt install aide

# Configurar monitoreo
sudo tee -a /etc/aide/aide.conf << EOF
/opt/aegis-framework/config f+p+u+g+s+m+c+md5+sha1
/opt/aegis-framework/.env f+p+u+g+s+m+c+md5+sha1
EOF

# Inicializar base de datos
sudo aide --init
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db
```

---

## 💾 Backup y Recuperación

### Estrategia de Backup

#### 1. Backup Automático Diario
```bash
# Crear script de backup automático
cat > scripts/daily-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/aegis"
DATE=$(date +%Y%m%d)
RETENTION_DAYS=30

# Crear backup
./scripts/backup-config.sh --backup-path "$BACKUP_DIR/daily-$DATE"

# Limpiar backups antiguos
find "$BACKUP_DIR" -name "daily-*" -mtime +$RETENTION_DAYS -delete

# Log resultado
echo "$(date): Backup completed - daily-$DATE" >> logs/backup.log
EOF

chmod +x scripts/daily-backup.sh
```

#### 2. Configurar Cron Job
```bash
# Agregar a crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/aegis-framework/scripts/daily-backup.sh") | crontab -
```

#### 3. Backup Remoto
```bash
# Sincronizar con servidor remoto
rsync -avz --delete /opt/backups/aegis/ user@backup-server:/backups/aegis/

# Usar rclone para cloud storage
rclone sync /opt/backups/aegis/ remote:aegis-backups/
```

### Procedimientos de Recuperación

#### 1. Recuperación Completa
```bash
# Detener servicios
./scripts/stop-all-services.sh --force

# Restaurar desde backup
./scripts/backup-config.sh --restore backup-20240101-120000

# Verificar integridad
./scripts/health-check.sh --detailed

# Reiniciar servicios
./scripts/start-all-services.sh
```

#### 2. Recuperación Parcial
```bash
# Restaurar solo configuración
cp backups/latest/.env .
cp backups/latest/config/app_config.json config/

# Restaurar solo datos
cp -r backups/latest/data/ .

# Reiniciar servicios afectados
./scripts/restart-service.sh dashboard
```

#### 3. Migración a Nuevo Servidor
```bash
# En servidor origen
./scripts/backup-config.sh --full

# Transferir backup
scp -r backups/full-backup-* user@new-server:/tmp/

# En servidor destino
./scripts/install-dependencies.sh
./scripts/restore-from-backup.sh /tmp/full-backup-*
./scripts/health-check.sh
```

### Verificación de Backups

#### Script de Verificación
```bash
cat > scripts/verify-backup.sh << 'EOF'
#!/bin/bash
BACKUP_PATH=$1

if [[ -z "$BACKUP_PATH" ]]; then
    echo "Uso: $0 <ruta_backup>"
    exit 1
fi

echo "Verificando backup: $BACKUP_PATH"

# Verificar archivos críticos
CRITICAL_FILES=(".env" "config/app_config.json" "config/torrc")

for file in "${CRITICAL_FILES[@]}"; do
    if [[ -f "$BACKUP_PATH/$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file - FALTANTE"
    fi
done

# Verificar integridad de archivos
if command -v sha256sum &> /dev/null; then
    find "$BACKUP_PATH" -type f -exec sha256sum {} \; > "$BACKUP_PATH.sha256"
    echo "✅ Checksums generados"
fi

echo "Verificación completada"
EOF

chmod +x scripts/verify-backup.sh
```

---

## 📚 Referencias Adicionales

### Documentación Técnica
- [API Documentation](API_REFERENCE.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Security Guidelines](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)

### Scripts de Utilidad
- `install-dependencies.ps1/sh` - Instalación de dependencias
- `setup-config.ps1/sh` - Configuración inicial
- `start-all-services.ps1/sh` - Iniciar todos los servicios
- `stop-all-services.ps1/sh` - Detener todos los servicios
- `health-check.ps1/sh` - Verificación de salud del sistema
- `monitor-services.ps1/sh` - Monitoreo continuo
- `backup-config.ps1/sh` - Gestión de backups
- `update-system.ps1/sh` - Actualización del sistema
- `deploy-production.ps1/sh` - Despliegue en producción

### Recursos Externos
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Node.js Best Practices](https://nodejs.org/en/docs/guides/)
- [Tor Configuration](https://www.torproject.org/docs/tor-manual.html)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/getting-started/)

### Soporte y Comunidad
- **Issues**: [GitHub Issues](https://github.com/your-org/aegis-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/aegis-framework/discussions)
- **Wiki**: [Project Wiki](https://github.com/your-org/aegis-framework/wiki)
- **Email**: support@aegis-framework.org

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, lee la [Guía de Contribución](CONTRIBUTING.md) antes de enviar pull requests.

---

**© 2024 AEGIS Framework Team. Todos los derechos reservados.**