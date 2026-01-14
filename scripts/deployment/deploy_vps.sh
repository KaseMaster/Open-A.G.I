#!/bin/bash
# Script de Despliegue AEGIS Framework en VPS
# Programador Principal: Jose Gómez alias KaseMaster
# Contacto: kasemaster@aegis-framework.com
# Versión: 2.0.0
# Licencia: MIT

set -e

# Configuración del VPS
VPS_IP="77.237.235.224"
VPS_USER="root"
PROJECT_DIR="/opt/openagi"
SERVICE_USER="openagi"

echo "🚀 Iniciando despliegue de AEGIS Framework en VPS $VPS_IP"

# Función para logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Actualizar sistema
log "Actualizando sistema..."
apt update && apt upgrade -y

# Instalar dependencias del sistema
log "Instalando dependencias del sistema..."
apt install -y python3 python3-pip python3-venv git nginx supervisor redis-server \
    postgresql postgresql-contrib tor curl wget unzip build-essential \
    libssl-dev libffi-dev python3-dev pkg-config

# Crear usuario del servicio
log "Creando usuario del servicio..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d $PROJECT_DIR $SERVICE_USER
fi

# Crear directorio del proyecto
log "Creando directorio del proyecto..."
mkdir -p $PROJECT_DIR
chown $SERVICE_USER:$SERVICE_USER $PROJECT_DIR

# Configurar PostgreSQL
log "Configurando PostgreSQL..."
sudo -u postgres createuser --createdb $SERVICE_USER || true
sudo -u postgres createdb openagi_db -O $SERVICE_USER || true

# Configurar Redis
log "Configurando Redis..."
systemctl enable redis-server
systemctl start redis-server

# Configurar Tor
log "Configurando Tor..."
systemctl enable tor
systemctl start tor

# Configurar firewall
log "Configurando firewall..."
ufw allow ssh
ufw allow 80
ufw allow 443
ufw allow 8051  # Dashboard
ufw allow 8086  # Chat PHP
ufw allow 9050  # Tor SOCKS
ufw --force enable

log "✅ Configuración base del VPS completada"