# AEGIS Framework - Guía de Despliegue de Producción

## 📋 Resumen

Esta guía describe el proceso completo para desplegar el AEGIS Framework en un entorno de producción con configuración WSGI/ASGI optimizada.

## 🚀 Componentes del Despliegue

### Scripts de Despliegue Creados

1. **`deploy_wsgi_simple.py`** - Script Python simplificado para despliegue WSGI
2. **`deploy_production_complete.py`** - Script Python completo con monitoreo y auto-recuperación
3. **`production_start.sh`** - Script Bash para Linux/Unix con gestión completa de servicios
4. **`production_start.bat`** - Script Batch para Windows
5. **`gunicorn_config.py`** - Configuración avanzada de Gunicorn
6. **`uvicorn_config.py`** - Configuración avanzada de Uvicorn
7. **`wsgi_server_manager.py`** - Gestor de servidores WSGI/ASGI

## 📦 Requisitos Previos

### Sistema Operativo
- Linux (Ubuntu 20.04+, CentOS 8+, Debian 10+)
- Windows Server 2019+
- macOS (para desarrollo)

### Python
- Python 3.8 o superior
- pip (gestor de paquetes)

### Módulos Python Requeridos
```bash
pip install gunicorn uvicorn flask fastapi redis psycopg2-binary
```

### Dependencias del Sistema (Linux)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv nginx redis-server postgresql

# CentOS/RHEL
sudo yum install python3-pip python3-venv nginx redis postgresql-server
```

## 🔧 Configuración de Producción

### 1. Preparar el Entorno

```bash
# Crear directorio de proyecto
mkdir -p /opt/aegis
cd /opt/aegis

# Copiar archivos del proyecto
cp -r /ruta/a/tu/proyecto/* .

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Crear archivo `.env` con:
```bash
# Configuración general
AEGIS_ENVIRONMENT=production
AEGIS_LOG_LEVEL=info
AEGIS_SECRET_KEY=tu_clave_secreta_aqui

# Base de datos
DATABASE_URL=postgresql://usuario:password@localhost/aegis_db
REDIS_URL=redis://localhost:6379/0

# Seguridad
JWT_SECRET_KEY=tu_jwt_secreto
ENCRYPTION_KEY=tu_clave_encriptacion

# Monitoreo
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
```

### 3. Configurar Base de Datos

```bash
# PostgreSQL
sudo -u postgres psql
CREATE DATABASE aegis_db;
CREATE USER aegis_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE aegis_db TO aegis_user;
\q

# Redis (ya debería estar corriendo)
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## 🚀 Despliegue con Scripts

### Opción 1: Script Python Completo (Recomendado)

```bash
# Hacer ejecutable el script
chmod +x deploy_production_complete.py

# Ejecutar despliegue
python3 deploy_production_complete.py
```

Este script:
- ✅ Realiza verificaciones previas
- ✅ Configura todos los servicios
- ✅ Inicia monitoreo automático
- ✅ Proporciona auto-recuperación
- ✅ Genera reportes de despliegue

### Opción 2: Script Bash

```bash
# Hacer ejecutable
chmod +x production_start.sh

# Iniciar servicios
./production_start.sh start

# Ver estado
./production_start.sh status

# Ver logs
./production_start.sh logs aegis-node
```

### Opción 3: Script Python Simplificado

```bash
python3 deploy_wsgi_simple.py
```

## 🔍 Verificación del Despliegue

### Verificar Servicios

```bash
# Verificar que los puertos estén escuchando
netstat -tlnp | grep -E ':(8080|8000|3000|8081)'

# O con ss
ss -tlnp | grep -E ':(8080|8000|3000|8081)'
```

### Verificar Logs

```bash
# Logs generales
tail -f logs/deploy.log

# Logs específicos por servicio
tail -f logs/node.log
tail -f logs/api.log
tail -f logs/dashboard.log
```

### Pruebas de Salud

```bash
# Verificar endpoints de salud
curl http://localhost:8080/health
curl http://localhost:8000/health
curl http://localhost:3000/health
```

## 🔒 Configuración de Seguridad

### Firewall (UFW - Ubuntu)

```bash
# Habilitar firewall
sudo ufw enable

# Permitir SSH
sudo ufw allow 22/tcp

# Permitir servicios AEGIS
sudo ufw allow 8080/tcp  # Node
sudo ufw allow 8000/tcp  # API
sudo ufw allow 3000/tcp  # Dashboard
sudo ufw allow 8081/tcp  # Admin

# Ver estado
sudo ufw status
```

### Configuración Nginx como Reverse Proxy

Crear archivo `/etc/nginx/sites-available/aegis`:
```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /node {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/aegis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL/TLS con Let's Encrypt

```bash
# Instalar Certbot
sudo apt install certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d tu-dominio.com

# Configurar renovación automática
sudo crontab -e
# Agregar: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoreo y Mantenimiento

### Monitoreo con Systemd

Crear servicios systemd personalizados:

`/etc/systemd/system/aegis-node.service`:
```ini
[Unit]
Description=AEGIS Node Service
After=network.target

[Service]
Type=forking
User=aegis
Group=aegis
WorkingDirectory=/opt/aegis
ExecStart=/opt/aegis/production_start.sh start
ExecStop=/opt/aegis/production_start.sh stop
ExecReload=/opt/aegis/production_start.sh restart
PIDFile=/opt/aegis/pids/aegis-node.pid
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Habilitar servicio
sudo systemctl enable aegis-node
sudo systemctl start aegis-node
sudo systemctl status aegis-node
```

### Monitoreo con Supervisor

Instalar y configurar Supervisor:

```bash
# Instalar
sudo apt install supervisor

# Crear configuración
sudo nano /etc/supervisor/conf.d/aegis.conf
```

```ini
[program:aegis-node]
command=python3 /opt/aegis/deploy_wsgi_simple.py
directory=/opt/aegis
user=aegis
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/aegis/logs/supervisor-node.log
environment=AEGIS_ENVIRONMENT="production"
```

```bash
# Recargar configuración
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status
```

### Alertas y Notificaciones

Configurar alertas para:
- 🔴 Servicios caídos
- 🟡 Alto uso de CPU/memoria
- 🟠 Errores en logs
- 🔵 Actualizaciones de seguridad

## 🔄 Actualizaciones y Mantenimiento

### Proceso de Actualización

1. **Backup del sistema**
```bash
# Backup de configuración
cp -r /opt/aegis /opt/aegis.backup.$(date +%Y%m%d)

# Backup de base de datos
sudo -u postgres pg_dump aegis_db > aegis_backup_$(date +%Y%m%d).sql
```

2. **Actualizar código**
```bash
cd /opt/aegis
git pull origin main
```

3. **Actualizar dependencias**
```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

4. **Reiniciar servicios**
```bash
./production_start.sh restart
```

### Mantenimiento Programado

- **Diario**: Verificar logs, espacio en disco
- **Semanal**: Actualizar dependencias de seguridad
- **Mensual**: Backup completo, análisis de rendimiento
- **Trimestral**: Auditoría de seguridad

## 🚨 Solución de Problemas

### Servicios No Inician

1. Verificar logs:
```bash
tail -n 50 logs/node.log
```

2. Verificar puertos:
```bash
netstat -tlnp | grep 8080
```

3. Verificar permisos:
```bash
ls -la /opt/aegis
```

### Alto Uso de Recursos

1. Verificar procesos:
```bash
top -p $(pgrep -f "gunicorn\|uvicorn")
```

2. Verificar conexiones:
```bash
ss -tuln | grep -E ':(8080|8000|3000)'
```

3. Ajustar workers en configuración

### Errores de Base de Datos

1. Verificar conexión PostgreSQL:
```bash
sudo -u postgres psql -c "SELECT 1"
```

2. Verificar Redis:
```bash
redis-cli ping
```

## 📞 Soporte y Contacto

Para soporte técnico:
- 📧 Email: soporte@protonmail.com
- 💬 Discord: [AEGIS Community](https://discord.gg/aegis)
- 📚 Documentación: [docs.protonmail.com](https://docs.protonmail.com)
- 🐛 Issues: [GitHub Issues](https://github.com/aegis-framework/aegis/issues)

## 📄 Licencia y Avisos Legales

Este framework se distribuye bajo licencia MIT. Asegúrate de:
- ✅ Cumplir con las leyes locales de protección de datos
- ✅ Implementar auditorías de seguridad regulares
- ✅ Mantener logs de acceso según regulaciones
- ✅ Actualizar parches de seguridad

---

**⚠️ IMPORTANTE**: Esta guía es para entornos de producción. Siempre realiza pruebas en entornos de desarrollo/staging antes de desplegar en producción.