#!/bin/bash
# Script de despliegue mejorado para AEGIS Framework en VPS
# Implementa todas las recomendaciones de seguridad y optimización
# Autor: AEGIS Security Team
# Versión: 3.0.0

set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuración
AEGIS_USER="aegis"
AEGIS_HOME="/home/$AEGIS_USER"
AEGIS_DIR="$AEGIS_HOME/openagi"
LOG_FILE="/var/log/aegis-deploy.log"
BACKUP_DIR="/var/backups/aegis"
CONFIG_FILE="$AEGIS_DIR/config/production_config.json"

# Funciones de logging
log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR${NC} $1" | tee -a "$LOG_FILE"
}

log_header() {
    echo -e "\n${PURPLE}$(printf '=%.0s' {1..80})${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}  $1${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}$(printf '=%.0s' {1..80})${NC}\n" | tee -a "$LOG_FILE"
}

# Verificaciones iniciales
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Este script debe ejecutarse como root"
        exit 1
    fi
}

check_internet() {
    log_info "Verificando conectividad a Internet..."
    if ! ping -c 1 8.8.8.8 &> /dev/null; then
        log_error "No hay conectividad a Internet"
        exit 1
    fi
    
    # Verificar DNS
    if ! nslookup google.com &> /dev/null; then
        log_error "DNS no está funcionando correctamente"
        exit 1
    fi
    
    log_success "Conectividad verificada"
}

# Sistema de respaldo
create_backup_system() {
    log_header "CONFIGURANDO SISTEMA DE RESPALDO"
    
    mkdir -p "$BACKUP_DIR"/{configs,databases,logs}
    
    # Script de respaldo automático
    cat > /usr/local/bin/aegis-backup << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/aegis"
DATE=$(date +%Y%m%d_%H%M%S)

# Respaldo de configuraciones
tar -czf "$BACKUP_DIR/configs/config_$DATE.tar.gz" /etc/nginx /etc/supervisor /etc/tor /etc/fail2ban

# Respaldo de bases de datos
sudo -u postgres pg_dump openagi_db > "$BACKUP_DIR/databases/db_$DATE.sql"

# Respaldo de logs
tar -czf "$BACKUP_DIR/logs/logs_$DATE.tar.gz" /var/log/aegis* /var/log/nginx /var/log/tor

# Limpiar respaldos antiguos (mantener últimos 7 días)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR/databases" -name "*.sql" -mtime +7 -delete
EOF
    
    chmod +x /usr/local/bin/aegis-backup
    
    # Configurar cron para respaldos automáticos
    echo "0 2 * * * root /usr/local/bin/aegis-backup" >> /etc/crontab
    
    log_success "Sistema de respaldo configurado"
}

# Seguridad del sistema
harden_system() {
    log_header "ENDURECIENDO SISTEMA"
    
    # Actualizar sistema
    log_info "Actualizando sistema..."
    apt-get update -y
    apt-get upgrade -y
    apt-get dist-upgrade -y
    apt-get autoremove -y
    apt-get autoclean -y
    
    # Instalar herramientas de seguridad
    apt-get install -y \
        fail2ban \
        ufw \
        unattended-upgrades \
        apt-listchanges \
        needrestart \
        debsecan \
        lynis \
        chkrootkit \
        rkhunter \
        aide \
        apparmor \
        apparmor-profiles \
        apparmor-utils
    
    # Configurar actualizaciones automáticas
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
        "\${distro_id}:\${distro_codename}-security";
        "\${distro_id}ESMApps:\${distro_codename}-apps-security";
        "\${distro_id}ESM:\${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF
    
    # Configurar AppArmor
    systemctl enable apparmor
    systemctl start apparmor
    
    # Configurar kernel parameters
    cat > /etc/sysctl.d/99-aegis-security.conf << 'EOF'
# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_timestamps = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# IPv6 security (disable if not needed)
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1

# Kernel security
kernel.randomize_va_space = 2
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.core_uses_pid = 1
kernel.sysrq = 0
EOF
    
    sysctl -p /etc/sysctl.d/99-aegis-security.conf
    
    log_success "Sistema endurecido"
}

# Firewall avanzado
setup_advanced_firewall() {
    log_header "CONFIGURANDO FIREWALL AVANZADO"
    
    # Resetear UFW
    ufw --force reset
    
    # Políticas por defecto
    ufw default deny incoming
    ufw default allow outgoing
    ufw default deny routed
    
    # SSH seguro (cambiar puerto si es necesario)
    ufw allow 22/tcp
    
    # Servicios web
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # AEGIS Framework
    ufw allow 8080/tcp comment "AEGIS Dashboard"
    ufw allow 8051/tcp comment "AEGIS Monitoring"
    ufw allow 8000/tcp comment "AEGIS API"
    ufw allow 9050/tcp comment "TOR SOCKS"
    ufw allow 9051/tcp comment "TOR Control"
    
    # Rate limiting
    ufw limit 22/tcp
    
    # Habilitar firewall
    ufw --force enable
    
    # Verificar estado
    ufw status verbose
    
    log_success "Firewall configurado"
}

# Fail2ban avanzado
setup_advanced_fail2ban() {
    log_header "CONFIGURANDO FAIL2BAN AVANZADO"
    
    cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
# Tiempo de bloqueo: 1 hora
bantime = 3600
# Tiempo para buscar intentos: 10 minutos
findtime = 600
# Máximo de reintentos: 5
maxretry = 5
# Ignorar redes locales
ignoreip = 127.0.0.1/8 ::1 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16

# SSH
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

# Nginx
[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-noscript]
enabled = true
filter = nginx-noscript
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 6

[nginx-badbots]
enabled = true
filter = nginx-badbots
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-noproxy]
enabled = true
filter = nginx-noproxy
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 2
EOF
    
    # Filtros personalizados
    cat > /etc/fail2ban/filter.d/nginx-badbots.conf << 'EOF'
[Definition]
badbotscustom = EmailCollector|WebEMailExtrac|TrackBack/1\.02|sogou music spider
failregex = ^<HOST> -.*"(GET|POST|HEAD).*".*"(?:%(badbotscustom)s|%(badbots)s)".*$
ignoreregex =
EOF
    
    cat > /etc/fail2ban/filter.d/nginx-noproxy.conf << 'EOF'
[Definition]
failregex = ^<HOST> -.*"(GET|POST|HEAD).*".*".*".*"(?!http://|https://).*$"
ignoreregex =
EOF
    
    systemctl enable fail2ban
    systemctl restart fail2ban
    
    log_success "Fail2ban configurado"
}

# Instalación de dependencias
install_dependencies() {
    log_header "INSTALANDO DEPENDENCIAS"
    
    # Actualizar repositorios
    apt-get update -y
    
    # Dependencias del sistema
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        pkg-config \
        git \
        curl \
        wget \
        unzip \
        tar \
        htop \
        net-tools \
        nmap \
        tcpdump \
        strace \
        ltrace \
        gdb \
        valgrind \
        nginx \
        supervisor \
        redis-server \
        postgresql \
        postgresql-contrib \
        postgresql-server-dev-all \
        tor \
        torsocks \
        privoxy \
        tor-arm \
        certbot \
        python3-certbot-nginx \
        python3-certbot-dns-cloudflare \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        libfreetype6-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        python3-openssl
    
    log_success "Dependencias instaladas"
}

# Crear usuario AEGIS seguro
create_secure_user() {
    log_header "CREANDO USUARIO AEGIS SEGURO"
    
    if id "$AEGIS_USER" &>/dev/null; then
        log_warning "Usuario $AEGIS_USER ya existe"
    else
        # Crear usuario con directorio home y bash
        useradd -m -s /bin/bash -d "$AEGIS_HOME" "$AEGIS_USER"
        
        # Configurar límites de recursos
        cat > /etc/security/limits.d/aegis.conf << EOF
# Límites para usuario AEGIS
aegis    soft    nproc       4096
aegis    hard    nproc       8192
aegis    soft    nofile      65536
aegis    hard    nofile      65536
aegis    soft    fsize       unlimited
aegis    hard    fsize       unlimited
aegis    soft    cpu         unlimited
aegis    hard    cpu         unlimited
aegis    soft    as          unlimited
aegis    hard    as          unlimited
EOF
        
        # Configurar sudo seguro
        echo "$AEGIS_USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nginx, /usr/bin/systemctl restart supervisor, /usr/bin/supervisorctl restart all" >> /etc/sudoers.d/aegis
        chmod 440 /etc/sudoers.d/aegis
        
        log_success "Usuario $AEGIS_USER creado con seguridad mejorada"
    fi
}

# Configurar SSH seguro
setup_secure_ssh() {
    log_header "CONFIGURANDO SSH SEGURO"
    
    # Backup de configuración SSH
    cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup.$(date +%Y%m%d_%H%M%S)
    
    # Configuración SSH segura
    cat > /etc/ssh/sshd_config << 'EOF'
# SSH Configuration for AEGIS Framework
Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_ed25519_key

# Authentication
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes

# Security
X11Forwarding no
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server

# Connection limits
MaxAuthTries 3
MaxSessions 2
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2

# Ciphers and algorithms (secure)
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,hmac-sha2-256,hmac-sha2-512
KexAlgorithms curve25519-sha256@libssh.org,diffie-hellman-group16-sha512,diffie-hellman-group18-sha512
EOF
    
    # Reiniciar SSH
    systemctl restart sshd
    
    # Verificar configuración
    sshd -t
    
    log_success "SSH configurado con seguridad mejorada"
}

# Configurar PostgreSQL seguro
setup_secure_postgresql() {
    log_header "CONFIGURANDO POSTGRESQL SEGURO"
    
    # Iniciar PostgreSQL
    systemctl enable postgresql
    systemctl start postgresql
    
    # Esperar a que PostgreSQL esté listo
    sleep 5
    
    # Configuración de seguridad PostgreSQL
    cat > /etc/postgresql/*/main/pg_hba.conf << 'EOF'
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
local   all             aegis                                   md5
host    all             aegis           127.0.0.1/32            md5
host    all             aegis           ::1/128                 md5
host    all             all             0.0.0.0/0               reject
host    all             all             ::/0                    reject
EOF
    
    # Configuración principal PostgreSQL
    cat > /etc/postgresql/*/main/postgresql.conf << 'EOF'
# Security settings
ssl = on
ssl_cert_file = '/etc/ssl/certs/ssl-cert-snakeoil.pem'
ssl_key_file = '/etc/ssl/private/ssl-cert-snakeoil.key'
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
ssl_prefer_server_ciphers = on
password_encryption = scram-sha-256

# Connection settings
listen_addresses = 'localhost'
port = 5432
max_connections = 100
superuser_reserved_connections = 3

# Logging settings
log_destination = 'stderr'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
EOF
    
    # Crear usuario y base de datos
    sudo -u postgres psql << EOF
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'aegis') THEN
        CREATE ROLE aegis WITH LOGIN PASSWORD '$(openssl rand -base64 32)';
    END IF;
END
\$\$;

DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aegis_db') THEN
        CREATE DATABASE aegis_db OWNER aegis;
    END IF;
END
\$\$;

GRANT ALL PRIVILEGES ON DATABASE aegis_db TO aegis;
EOF
    
    # Reiniciar PostgreSQL
    systemctl restart postgresql
    
    log_success "PostgreSQL configurado con seguridad mejorada"
}

# Configurar Redis seguro
setup_secure_redis() {
    log_header "CONFIGURANDO REDIS SEGURO"
    
    # Configuración de seguridad Redis
    cat > /etc/redis/redis.conf << 'EOF'
# Network
bind 127.0.0.1
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300

# Security
requirepass $(openssl rand -base64 32)
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
rename-command CONFIG ""
rename-command SHUTDOWN AEGIS_SHUTDOWN

# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Limits
maxclients 100
EOF
    
    # Configurar AppArmor para Redis
    cat > /etc/apparmor.d/local/redis-server << 'EOF'
# Redis AppArmor profile
/var/lib/redis/** rwk,
/var/log/redis/** rw,
EOF
    
    # Reiniciar servicios
    systemctl enable redis-server
    systemctl restart redis-server
    
    log_success "Redis configurado con seguridad mejorada"
}

# Configurar TOR seguro
setup_secure_tor() {
    log_header "CONFIGURANDO TOR SEGURO"
    
    # Configuración de TOR
    cat > /etc/tor/torrc << 'EOF'
# Basic configuration
ControlPort 9051
CookieAuthentication 1
CookieAuthFileGroupReadable 1
DataDirectory /var/lib/tor
PidFile /run/tor/tor.pid
RunAsDaemon 1
User debian-tor

# Logging
Log notice file /var/log/tor/notices.log
Log info file /var/log/tor/info.log

# Security
AvoidDiskWrites 1
DisableDebuggerAttachment 1
MaxCircuitDirtiness 600
NewCircuitPeriod 30
CircuitBuildTimeout 60

# Bandwidth
BandwidthRate 1 MB
BandwidthBurst 2 MB
MaxOnionQueueDelay 1750

# Hidden service for AEGIS
HiddenServiceDir /var/lib/tor/aegis/
HiddenServicePort 80 127.0.0.1:8080
HiddenServicePort 443 127.0.0.1:8443
EOF
    
    # Crear directorios
    mkdir -p /var/lib/tor/aegis
    chown debian-tor:debian-tor /var/lib/tor/aegis
    chmod 700 /var/lib/tor/aegis
    
    # Configurar AppArmor para TOR
    cat > /etc/apparmor.d/local/tor << 'EOF'
# TOR AppArmor profile
/var/lib/tor/** rwk,
/var/log/tor/** rw,
EOF
    
    # Reiniciar TOR
    systemctl enable tor
    systemctl restart tor
    
    # Esperar a que TOR cree el servicio oculto
    sleep 10
    
    # Obtener dirección onion
    if [ -f /var/lib/tor/aegis/hostname ]; then
        ONION_ADDRESS=$(cat /var/lib/tor/aegis/hostname)
        log_success "Servicio oculto TOR configurado: $ONION_ADDRESS"
    else
        log_warning "No se pudo obtener la dirección onion de TOR"
    fi
    
    log_success "TOR configurado con seguridad mejorada"
}

# Configurar Nginx con seguridad
setup_secure_nginx() {
    log_header "CONFIGURANDO NGINX CON SEGURIDAD"
    
    # Configuración principal de Nginx
    cat > /etc/nginx/nginx.conf << 'EOF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    # Basic Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;

    # Security
    client_max_body_size 16M;
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    
    # Gzip Settings
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    limit_req_zone $binary_remote_addr zone=general:10m rate=100r/s;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOF
    
    # Configuración del sitio AEGIS
    cat > /etc/nginx/sites-available/aegis << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' ws: wss:;";
    
    # Hide server version
    server_tokens off;
    
    # Rate limiting
    limit_req zone=general burst=100 nodelay;
    
    # Main dashboard
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer sizes
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Monitoring (restricted)
    location /monitoring {
        allow 127.0.0.1;
        allow ::1;
        deny all;
        
        proxy_pass http://127.0.0.1:8051;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }
    
    # API endpoints
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api burst=50 nodelay;
        
        # Security
        limit_req_status 429;
        proxy_hide_header X-Powered-By;
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # Block sensitive files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location ~ ~$ {
        deny all;
        access_log off;
        log_not_found off;
    }
    
    location ~* \.(conf|log|ini|sh|sql|bak|backup|old|tmp)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
}
EOF
    
    # Habilitar sitio
    ln -sf /etc/nginx/sites-available/aegis /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test y reiniciar Nginx
    nginx -t && systemctl restart nginx
    
    log_success "Nginx configurado con seguridad mejorada"
}

# Configurar Supervisor con seguridad
setup_secure_supervisor() {
    log_header "CONFIGURANDO SUPERVISOR CON SEGURIDAD"
    
    # Configuración de Supervisor
    cat > /etc/supervisor/supervisord.conf << 'EOF'
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0700
chown=root:root

[supervisord]
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor
user=root
nodaemon=false
minfds=1024
minprocs=200
loglevel=info

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[include]
files = /etc/supervisor/conf.d/*.conf
EOF
    
    # Configuración de servicios AEGIS
    cat > /etc/supervisor/conf.d/aegis.conf << 'EOF'
[program:aegis-node]
command=/home/aegis/openagi/venv/bin/python /home/aegis/openagi/main.py start-node --config /home/aegis/openagi/config/production_config.json
directory=/home/aegis/openagi
user=aegis
autostart=true
autorestart=true
startretries=3
startsecs=10
stopwaitsecs=10
redirect_stderr=true
stdout_logfile=/var/log/aegis-node.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
stdout_capture_maxbytes=50MB
environment=PATH="/home/aegis/openagi/venv/bin",PYTHONPATH="/home/aegis/openagi",HOME="/home/aegis",USER="aegis"

[program:aegis-dashboard]
command=/home/aegis/openagi/venv/bin/python /home/aegis/openagi/main.py start-dashboard --type web --port 8080 --config /home/aegis/openagi/config/production_config.json
directory=/home/aegis/openagi
user=aegis
autostart=true
autorestart=true
startretries=3
startsecs=10
stopwaitsecs=10
redirect_stderr=true
stdout_logfile=/var/log/aegis-dashboard.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
stdout_capture_maxbytes=50MB
environment=PATH="/home/aegis/openagi/venv/bin",PYTHONPATH="/home/aegis/openagi",HOME="/home/aegis",USER="aegis"

[program:aegis-wsgi]
command=/home/aegis/openagi/venv/bin/python /home/aegis/openagi/wsgi_production.py --host 127.0.0.1 --port 8000 --workers 4
directory=/home/aegis/openagi
user=aegis
autostart=true
autorestart=true
startretries=3
startsecs=10
stopwaitsecs=10
redirect_stderr=true
stdout_logfile=/var/log/aegis-wsgi.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
stdout_capture_maxbytes=50MB
environment=PATH="/home/aegis/openagi/venv/bin",PYTHONPATH="/home/aegis/openagi",HOME="/home/aegis",USER="aegis"
EOF
    
    # Recargar Supervisor
    systemctl enable supervisor
    systemctl restart supervisor
    
    # Esperar a que inicie
    sleep 5
    
    # Verificar estado
    supervisorctl status
    
    log_success "Supervisor configurado con seguridad mejorada"
}

# Función principal de despliegue
main_deploy() {
    log_header "INICIANDO DESPLIEGUE SEGURO DE AEGIS FRAMEWORK"
    
    # Verificaciones iniciales
    check_root
    check_internet
    
    # Configuración de seguridad
    harden_system
    setup_advanced_firewall
    setup_advanced_fail2ban
    
    # Instalación de dependencias
    install_dependencies
    
    # Configuración de usuarios y servicios
    create_secure_user
    setup_secure_ssh
    setup_secure_postgresql
    setup_secure_redis
    setup_secure_tor
    setup_secure_nginx
    setup_secure_supervisor
    
    # Sistema de respaldo
    create_backup_system
    
    # Generar reporte final
    generate_deployment_report
    
    log_success "✅ Despliegue seguro de AEGIS Framework completado!"
    
    echo -e "\n${GREEN}🎉 AEGIS Framework está listo para usar!${NC}"
    echo -e "${BLUE}Dashboard:${NC} http://$(curl -s ifconfig.me):8080"
    echo -e "${BLUE}Monitoring:${NC} http://$(curl -s ifconfig.me):8051"
    echo -e "${BLUE}API:${NC} http://$(curl -s ifconfig.me):8000"
    
    if [ -f /var/lib/tor/aegis/hostname ]; then
        echo -e "${PURPLE}Onion Service:${NC} http://$(cat /var/lib/tor/aegis/hostname)"
    fi
    
    echo -e "${YELLOW}⚠️  IMPORTANTE:${NC}"
    echo -e "  - Revisa el archivo de configuración: $CONFIG_FILE"
    echo -e "  - Los logs están en: /var/log/"
    echo -e "  - El respaldo se ejecuta automáticamente cada día a las 2 AM"
    echo -e "  - Usa 'supervisorctl status' para verificar los servicios"
}

# Generar reporte de despliegue
generate_deployment_report() {
    log_header "GENERANDO REPORTE DE DESPLIEGUE"
    
    REPORT_FILE="/var/log/aegis-deployment-report.txt"
    
    cat > "$REPORT_FILE" << EOF
AEGIS Framework Deployment Report
Generated: $(date)
Server: $(hostname)
IP: $(curl -s ifconfig.me 2>/dev/null || echo "N/A")

SERVICES STATUS:
$(systemctl status postgresql redis-server tor nginx supervisor --no-pager -l 2>/dev/null || echo "Service status unavailable")

FIREWALL STATUS:
$(ufw status verbose 2>/dev/null || echo "UFW status unavailable")

FAIL2BAN STATUS:
$(fail2ban-client status 2>/dev/null || echo "Fail2ban status unavailable")

ONION SERVICE:
$(cat /var/lib/tor/aegis/hostname 2>/dev/null || echo "Onion service not configured")

SECURITY CONFIGURATION:
- SSH: Port 22, Root login disabled, Key auth only
- Firewall: UFW enabled with rate limiting
- Fail2ban: Active with custom filters
- SSL/TLS: Modern configuration
- AppArmor: Enabled
- Kernel hardening: Applied

BACKUP SYSTEM:
- Location: $BACKUP_DIR
- Schedule: Daily at 2 AM
- Retention: 7 days

LOG LOCATIONS:
- Application: /var/log/aegis-*.log
- Nginx: /var/log/nginx/
- System: /var/log/
- Supervisor: /var/log/supervisor/

EOF
    
    log_success "Reporte de despliegue generado: $REPORT_FILE"
}

# Función de limpieza
cleanup() {
    log_info "Limpiando archivos temporales..."
    apt-get autoremove -y
    apt-get autoclean
    rm -rf /tmp/* /var/tmp/*
    log_success "Limpieza completada"
}

# Manejo de errores
trap 'log_error "Error en línea $LINENO. Saliendo..."; exit 1' ERR

# Ejecutar despliegue
main_deploy

# Limpieza final
cleanup

log_success "🚀 Despliegue de AEGIS Framework completado exitosamente!"