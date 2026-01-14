#!/bin/bash

# Quantum Currency Production Deployment Script
# This script automates the deployment of Quantum Currency in a production environment
# with high availability, security hardening, and auto-healing capabilities.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="quantum-currency"
PROJECT_DIR="/opt/${PROJECT_NAME}"
USER_NAME="quantum_user"
GROUP_NAME="quantum_user"
SERVICE_NAME="quantum-gunicorn"
NGINX_CONFIG="/etc/nginx/sites-available/${PROJECT_NAME}"
NGINX_ENABLED="/etc/nginx/sites-enabled/${PROJECT_NAME}"
SYSTEMD_SERVICE="/etc/systemd/system/${SERVICE_NAME}.service"
SOCKET_PATH="/tmp/gunicorn.sock"
LOG_DIR="/var/log/${PROJECT_NAME}"
PROMETHEUS_RULES_DIR="/etc/prometheus/rules"
ALERTMANAGER_CONFIG="/etc/alertmanager/alertmanager.yml"

echo -e "${BLUE}🚀 Quantum Currency Production Deployment${NC}"
echo "=========================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root. Please run as a regular user with sudo privileges.${NC}" 
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

if ! command_exists pip3; then
    echo -e "${RED}pip3 is not installed. Please install pip3 first.${NC}"
    exit 1
fi

if ! command_exists nginx; then
    echo -e "${YELLOW}Nginx is not installed. Installing...${NC}"
    sudo apt update
    sudo apt install -y nginx
fi

if ! command_exists gunicorn; then
    echo -e "${YELLOW}Gunicorn is not installed. Installing...${NC}"
    pip3 install gunicorn
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Create dedicated user and group
echo -e "${BLUE}👤 Creating dedicated user and group...${NC}"
if ! id "${USER_NAME}" &>/dev/null; then
    sudo useradd --system --no-create-home --shell /bin/false ${USER_NAME}
    echo -e "${GREEN}✅ User ${USER_NAME} created${NC}"
else
    echo -e "${YELLOW}⚠️ User ${USER_NAME} already exists${NC}"
fi

# Create project directory
echo -e "${BLUE}📁 Setting up project directory...${NC}"
sudo mkdir -p ${PROJECT_DIR}
sudo chown ${USER_NAME}:${GROUP_NAME} ${PROJECT_DIR}
sudo chmod 755 ${PROJECT_DIR}

# Copy project files (assuming this script is run from the project root)
echo -e "${BLUE}📦 Copying project files...${NC}"
sudo cp -r . ${PROJECT_DIR}/
sudo chown -R ${USER_NAME}:${GROUP_NAME} ${PROJECT_DIR}
sudo chmod -R 755 ${PROJECT_DIR}

# Create log directory
echo -e "${BLUE}📝 Setting up log directory...${NC}"
sudo mkdir -p ${LOG_DIR}
sudo chown ${USER_NAME}:${GROUP_NAME} ${LOG_DIR}
sudo chmod 755 ${LOG_DIR}

# Install Python dependencies
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
cd ${PROJECT_DIR}
sudo -u ${USER_NAME} pip3 install -r requirements.txt

# Create Systemd service file
echo -e "${BLUE}⚙️ Creating Systemd service...${NC}"
sudo tee ${SYSTEMD_SERVICE} > /dev/null <<EOF
[Unit]
Description=Quantum Currency Gunicorn Service
After=network.target

[Service]
User=${USER_NAME}
Group=${GROUP_NAME}
WorkingDirectory=${PROJECT_DIR}
Environment="QC_ENV=Production"
Environment="QC_LAMBDA_BOUNDS=0.8,1.2"
ExecStart=/usr/local/bin/gunicorn \\
    --workers \$(nproc) \\
    --threads 2 \\
    --timeout 90 \\
    --bind unix:${SOCKET_PATH} \\
    src.api.main:app \\
    --log-level warning \\
    --log-file ${LOG_DIR}/gunicorn.log \\
    --access-logfile ${LOG_DIR}/access.log
ExecReload=/bin/kill -s HUP \$MAINPID
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✅ Systemd service created at ${SYSTEMD_SERVICE}${NC}"

# Create Nginx configuration
echo -e "${BLUE}🌐 Creating Nginx configuration...${NC}"
sudo tee ${NGINX_CONFIG} > /dev/null <<EOF
server {
    listen 80;
    server_name _;  # Replace with your domain name

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;

    location / {
        proxy_pass http://unix:${SOCKET_PATH};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://unix:${SOCKET_PATH};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Metrics endpoint
    location /metrics {
        proxy_pass http://unix:${SOCKET_PATH};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Logging
    access_log ${LOG_DIR}/nginx_access.log;
    error_log ${LOG_DIR}/nginx_error.log;

    # Security: Hide version
    server_tokens off;
}

# HTTPS server (if SSL is configured)
server {
    listen 443 ssl http2;
    server_name _;  # Replace with your domain name

    # SSL Configuration - Enhanced Security
    ssl_protocols TLSv1.3;
    ssl_prefer_server_ciphers off;
    
    # Modern Ciphers (ensures Forward Secrecy, PFS)
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
    
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1h;
    ssl_session_tickets off;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;

    location / {
        proxy_pass http://unix:${SOCKET_PATH};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://unix:${SOCKET_PATH};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Metrics endpoint
    location /metrics {
        proxy_pass http://unix:${SOCKET_PATH};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Logging
    access_log ${LOG_DIR}/nginx_access.log;
    error_log ${LOG_DIR}/nginx_error.log;

    # Security: Hide version
    server_tokens off;
}
EOF

echo -e "${GREEN}✅ Nginx configuration created at ${NGINX_CONFIG}${NC}"

# Enable Nginx site
echo -e "${BLUE}🔗 Enabling Nginx site...${NC}"
sudo ln -sf ${NGINX_CONFIG} ${NGINX_ENABLED}
sudo nginx -t
sudo systemctl reload nginx

echo -e "${GREEN}✅ Nginx site enabled${NC}"

# Set proper permissions for socket
echo -e "${BLUE}🔐 Setting socket permissions...${NC}"
sudo mkdir -p $(dirname ${SOCKET_PATH})
sudo chown ${USER_NAME}:www-data $(dirname ${SOCKET_PATH})
sudo chmod 775 $(dirname ${SOCKET_PATH})

echo -e "${GREEN}✅ Socket permissions set${NC}"

# Start and enable Systemd service
echo -e "${BLUE}🚀 Starting and enabling Systemd service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

# Check service status
if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
    echo -e "${GREEN}✅ ${SERVICE_NAME} service started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start ${SERVICE_NAME} service${NC}"
    sudo systemctl status ${SERVICE_NAME}
    exit 1
fi

# Setup Prometheus alerting rules (if Prometheus is installed)
if command_exists prometheus; then
    echo -e "${BLUE}📊 Setting up Prometheus alerting rules...${NC}"
    sudo mkdir -p ${PROMETHEUS_RULES_DIR}
    
    sudo tee ${PROMETHEUS_RULES_DIR}/quantum_currency_alerts.yml > /dev/null <<EOF
groups:
- name: quantum_currency_alerts
  rules:
  - alert: CoherenceDensityCritical
    expr: quantum_currency_c_t < 0.85
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Quantum Currency Coherence Density (C_hat) is critically low."
      description: "Triggering Attunement Agent for system reboot/recalibration."

  - alert: LambdaDriftWarning
    expr: quantum_currency_lambda_t > 1.2 or quantum_currency_lambda_t < 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Dynamic Lambda (λ(t)) is drifting out of safe bounds."
      description: "System may require soft recalibration via CLI tool."
EOF
    
    echo -e "${GREEN}✅ Prometheus alerting rules created${NC}"
else
    echo -e "${YELLOW}⚠️ Prometheus not found. Skipping alerting rules setup.${NC}"
fi

# Setup Systemd timer for auto-healing (if Alertmanager is configured)
echo -e "${BLUE}🔧 Setting up auto-healing Systemd timer...${NC}"
sudo tee /etc/systemd/system/quantum-healing.service > /dev/null <<EOF
[Unit]
Description=Quantum Currency Auto-Healing Service
After=${SERVICE_NAME}.service

[Service]
Type=oneshot
User=${USER_NAME}
Group=${GROUP_NAME}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/scripts/healing_script.sh
EOF

sudo tee /etc/systemd/system/quantum-healing.timer > /dev/null <<EOF
[Unit]
Description=Run Quantum Currency Auto-Healing Every 5 Minutes
Requires=quantum-healing.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Create healing script
sudo tee ${PROJECT_DIR}/scripts/healing_script.sh > /dev/null <<EOF
#!/bin/bash
# Quantum Currency Auto-Healing Script

PROJECT_DIR="${PROJECT_DIR}"
LOG_FILE="${LOG_DIR}/healing.log"

log_message() {
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$1" | tee -a \$LOG_FILE
}

# Check if service is running
if ! systemctl is-active --quiet ${SERVICE_NAME}; then
    log_message "Service is not running. Attempting restart..."
    systemctl restart ${SERVICE_NAME}
    if systemctl is-active --quiet ${SERVICE_NAME}; then
        log_message "Service restarted successfully"
    else
        log_message "Failed to restart service"
    fi
else
    # Check coherence metrics via health endpoint
    HEALTH_CHECK=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost/health)
    
    if [ "\$HEALTH_CHECK" -eq 200 ]; then
        # Get actual metrics
        LAMBDA_T=\$(curl -s http://localhost/health | grep -o '"lambda_t":[0-9.]*' | cut -d':' -f2)
        C_T=\$(curl -s http://localhost/health | grep -o '"c_t":[0-9.]*' | cut -d':' -f2)
        
        # Check if lambda_t is out of bounds
        if (( \$(echo "\$LAMBDA_T < 0.8 || \$LAMBDA_T > 1.2" | bc -l) )); then
            log_message "Lambda drift detected: \$LAMBDA_T. Triggering recalibration..."
            # Execute Lambda Attunement Tool (Option 14) via CLI
            python3 ${PROJECT_DIR}/scripts/lambda_attunement_tool.py recalibrate
        fi
        
        # Check if C_t is critically low
        if (( \$(echo "\$C_T < 0.85" | bc -l) )); then
            log_message "Critical coherence density: \$C_T. Triggering hard restart..."
            systemctl restart ${SERVICE_NAME}
        fi
    else
        log_message "Health check failed with code \$HEALTH_CHECK. Restarting service..."
        systemctl restart ${SERVICE_NAME}
    fi
fi
EOF

sudo chmod +x ${PROJECT_DIR}/scripts/healing_script.sh
sudo chown ${USER_NAME}:${GROUP_NAME} ${PROJECT_DIR}/scripts/healing_script.sh

# Enable and start the timer
sudo systemctl daemon-reload
sudo systemctl enable quantum-healing.timer
sudo systemctl start quantum-healing.timer

echo -e "${GREEN}✅ Auto-healing Systemd timer created and started${NC}"

# Final status check
echo -e "${BLUE}✅ Final status check...${NC}"
echo -e "${GREEN}Service Status:${NC}"
sudo systemctl status ${SERVICE_NAME} --no-pager -l

echo -e "${GREEN}Nginx Status:${NC}"
sudo systemctl status nginx --no-pager -l

echo -e "${GREEN}Auto-healing Timer Status:${NC}"
sudo systemctl status quantum-healing.timer --no-pager -l

echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}🎉 Quantum Currency Production Deployment Complete!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e "📝 Summary:"
echo -e "  - Project deployed to: ${PROJECT_DIR}"
echo -e "  - Service name: ${SERVICE_NAME}"
echo -e "  - Nginx config: ${NGINX_CONFIG}"
echo -e "  - Logs directory: ${LOG_DIR}"
echo -e "  - Socket path: ${SOCKET_PATH}"
echo -e ""
echo -e "🔄 To restart the service:"
echo -e "  sudo systemctl restart ${SERVICE_NAME}"
echo -e ""
echo -e "📊 To check service status:"
echo -e "  sudo systemctl status ${SERVICE_NAME}"
echo -e ""
echo -e "📋 To view logs:"
echo -e "  sudo journalctl -u ${SERVICE_NAME} -f"
echo -e ""
echo -e "🔧 Health check endpoints:"
echo -e "  http://localhost/health"
echo -e "  http://localhost/metrics"
echo -e ""
echo -e "🛡️ Security features enabled:"
echo -e "  - Dedicated service user (${USER_NAME})"
echo -e "  - TLS 1.3 only (when SSL configured)"
echo -e "  - A+ SSL ciphers"
echo -e "  - Security headers"
echo -e "  - Auto-healing via Systemd timer"
echo -e ""
echo -e "⚡ High Availability features:"
echo -e "  - Multi-worker Gunicorn setup"
echo -e "  - Graceful reload capability"
echo -e "  - Prometheus monitoring integration"
echo -e "  - Automated restart on failure"
echo -e ""
echo -e "${GREEN}Deployment completed successfully!${NC}"