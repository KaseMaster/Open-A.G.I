#!/bin/bash
# Script de monitoreo y mantenimiento para AEGIS Framework
# Proporciona funciones de monitoreo, mantenimiento y respaldo
# Autor: AEGIS Security Team
# Versión: 2.0.0

set -euo pipefail

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/aegis"
BACKUP_DIR="/var/backups/aegis"
ALERT_EMAIL="admin@aegis.local"
MAX_DISK_USAGE=85
MAX_MEMORY_USAGE=90
MAX_CPU_USAGE=95
MAX_LOAD_AVG=8.0

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Funciones de logging
log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO${NC} $1" | tee -a "$LOG_DIR/maintenance.log"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS${NC} $1" | tee -a "$LOG_DIR/maintenance.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING${NC} $1" | tee -a "$LOG_DIR/maintenance.log"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR${NC} $1" | tee -a "$LOG_DIR/maintenance.log"
}

# Verificar estado de servicios
check_services() {
    log_info "Verificando estado de servicios..."
    
    local services=("nginx" "redis-server" "postgresql" "tor" "supervisor")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            log_success "✅ $service está activo"
        else
            log_error "❌ $service está inactivo"
            failed_services+=("$service")
        fi
    done
    
    # Verificar procesos de AEGIS
    if pgrep -f "python.*main.py" > /dev/null; then
        log_success "✅ Procesos de AEGIS están activos"
    else
        log_error "❌ Procesos de AEGIS están inactivos"
        failed_services+=("aegis-processes")
    fi
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_error "Servicios fallidos: ${failed_services[*]}"
        send_alert "Servicios críticos inactivos" "Los siguientes servicios están inactivos: ${failed_services[*]}"
    fi
}

# Verificar uso de recursos
check_resources() {
    log_info "Verificando uso de recursos..."
    
    # Uso de disco
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt "$MAX_DISK_USAGE" ]; then
        log_error "⚠️  Uso de disco alto: ${disk_usage}%"
        send_alert "Uso de disco alto" "El uso de disco es del ${disk_usage}%"
    else
        log_success "💾 Uso de disco: ${disk_usage}%"
    fi
    
    # Uso de memoria
    local memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$memory_usage" -gt "$MAX_MEMORY_USAGE" ]; then
        log_error "⚠️  Uso de memoria alto: ${memory_usage}%"
        send_alert "Uso de memoria alto" "El uso de memoria es del ${memory_usage}%"
    else
        log_success "🧠 Uso de memoria: ${memory_usage}%"
    fi
    
    # Uso de CPU
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    if (( $(echo "$cpu_usage > $MAX_CPU_USAGE" | bc -l) )); then
        log_error "⚠️  Uso de CPU alto: ${cpu_usage}%"
        send_alert "Uso de CPU alto" "El uso de CPU es del ${cpu_usage}%"
    else
        log_success "⚡ Uso de CPU: ${cpu_usage}%"
    fi
    
    # Load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
    if (( $(echo "$load_avg > $MAX_LOAD_AVG" | bc -l) )); then
        log_error "⚠️  Load average alto: $load_avg"
        send_alert "Load average alto" "El load average es $load_avg"
    else
        log_success "📊 Load average: $load_avg"
    fi
}

# Verificar logs de errores
check_logs() {
    log_info "Verificando logs de errores..."
    
    # Buscar errores en logs de AEGIS
    local error_count=0
    local warning_count=0
    
    if [ -f "$LOG_DIR/aegis-node.log" ]; then
        local node_errors=$(grep -c "ERROR" "$LOG_DIR/aegis-node.log" 2>/dev/null || echo "0")
        local node_warnings=$(grep -c "WARNING" "$LOG_DIR/aegis-node.log" 2>/dev/null || echo "0")
        error_count=$((error_count + node_errors))
        warning_count=$((warning_count + node_warnings))
    fi
    
    if [ -f "$LOG_DIR/aegis-dashboard.log" ]; then
        local dashboard_errors=$(grep -c "ERROR" "$LOG_DIR/aegis-dashboard.log" 2>/dev/null || echo "0")
        local dashboard_warnings=$(grep -c "WARNING" "$LOG_DIR/aegis-dashboard.log" 2>/dev/null || echo "0")
        error_count=$((error_count + dashboard_errors))
        warning_count=$((warning_count + dashboard_warnings))
    fi
    
    if [ -f "$LOG_DIR/aegis-wsgi.log" ]; then
        local wsgi_errors=$(grep -c "ERROR" "$LOG_DIR/aegis-wsgi.log" 2>/dev/null || echo "0")
        local wsgi_warnings=$(grep -c "WARNING" "$LOG_DIR/aegis-wsgi-wsgi.log" 2>/dev/null || echo "0")
        error_count=$((error_count + wsgi_errors))
        warning_count=$((warning_count + wsgi_warnings))
    fi
    
    log_info "📋 Resumen de logs:"
    log_info "  Errores: $error_count"
    log_info "  Advertencias: $warning_count"
    
    if [ "$error_count" -gt 10 ]; then
        send_alert "Muchos errores en logs" "Se encontraron $error_count errores en los logs"
    fi
}

# Verificar conectividad de red
check_network() {
    log_info "Verificando conectividad de red..."
    
    # Verificar TOR
    if systemctl is-active --quiet tor; then
        if curl -s --socks5 127.0.0.1:9050 https://check.torproject.org/api/ip > /dev/null; then
            log_success "✅ TOR está funcionando correctamente"
        else
            log_error "❌ TOR no está funcionando correctamente"
            send_alert "TOR no funciona" "El servicio TOR no está respondiendo correctamente"
        fi
    fi
    
    # Verificar conectividad a Internet
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        log_success "✅ Conectividad a Internet verificada"
    else
        log_error "❌ Sin conectividad a Internet"
        send_alert "Sin Internet" "El servidor no tiene conectividad a Internet"
    fi
    
    # Verificar puertos abiertos
    local open_ports=$(netstat -tuln | grep LISTEN | wc -l)
    log_info "📡 Puertos abiertos: $open_ports"
}

# Verificar seguridad
check_security() {
    log_info "Verificando seguridad..."
    
    # Verificar actualizaciones de seguridad
    if apt-get -s upgrade | grep -q "security"; then
        log_warning "⚠️  Hay actualizaciones de seguridad disponibles"
    else
        log_success "✅ No hay actualizaciones de seguridad pendientes"
    fi
    
    # Verificar fail2ban
    if fail2ban-client status > /dev/null 2>&1; then
        local banned_ips=$(fail2ban-client status sshd | grep "Currently banned" | awk '{print $3}')
        if [ "$banned_ips" -gt 0 ]; then
            log_warning "⚠️  Hay $banned_ips IPs baneadas por fail2ban"
        else
            log_success "✅ No hay IPs baneadas por fail2ban"
        fi
    fi
    
    # Verificar intentos de login fallidos
    local failed_logins=$(grep "Failed password" /var/log/auth.log | wc -l)
    if [ "$failed_logins" -gt 50 ]; then
        log_warning "⚠️  Muchos intentos de login fallidos: $failed_logins"
        send_alert "Intentos de login sospechosos" "Se encontraron $failed_logins intentos de login fallidos"
    else
        log_success "✅ Intentos de login normales: $failed_logins"
    fi
}

# Función de respaldo
perform_backup() {
    log_info "Realizando respaldo completo..."
    
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/backup_$backup_date"
    
    mkdir -p "$backup_path"/{configs,databases,logs,application}
    
    # Respaldo de configuraciones
    log_info "Respaldo de configuraciones..."
    tar -czf "$backup_path/configs/etc_configs.tar.gz" \
        /etc/nginx \
        /etc/supervisor \
        /etc/tor \
        /etc/fail2ban \
        /etc/redis \
        /etc/postgresql \
        2>/dev/null || true
    
    # Respaldo de bases de datos
    log_info "Respaldo de bases de datos..."
    if systemctl is-active --quiet postgresql; then
        sudo -u postgres pg_dump aegis_db > "$backup_path/databases/aegis_db.sql" 2>/dev/null || true
    fi
    
    # Respaldo de logs
    log_info "Respaldo de logs..."
    tar -czf "$backup_path/logs/logs_backup.tar.gz" \
        /var/log/aegis* \
        /var/log/nginx \
        /var/log/tor \
        /var/log/redis \
        /var/log/postgresql \
        2>/dev/null || true
    
    # Respaldo de aplicación
    log_info "Respaldo de aplicación..."
    if [ -d "/home/aegis/openagi" ]; then
        tar -czf "$backup_path/application/aegis_app.tar.gz" \
            -C /home/aegis openagi \
            --exclude="*.log" \
            --exclude="*.pyc" \
            --exclude="__pycache__" \
            --exclude=".git" \
            2>/dev/null || true
    fi
    
    # Crear resumen del respaldo
    cat > "$backup_path/backup_summary.txt" << EOF
AEGIS Framework Backup Summary
Date: $(date)
Backup Path: $backup_path

Contents:
$(ls -la "$backup_path"/)

Disk Usage:
$(du -sh "$backup_path"/*)

System Info:
Hostname: $(hostname)
Uptime: $(uptime)
Load: $(uptime | awk -F'load average:' '{print $2}')
Disk Usage: $(df -h / | awk 'NR==2 {print $5}')
Memory Usage: $(free -h | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
EOF
    
    # Limpiar respaldos antiguos (mantener últimos 7 días)
    find "$BACKUP_DIR" -maxdepth 1 -type d -name "backup_*" -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    
    log_success "✅ Respaldo completado: $backup_path"
}

# Función de actualización
perform_update() {
    log_info "Realizando actualización de sistema..."
    
    # Actualizar paquetes
    apt-get update -y
    apt-get upgrade -y
    
    # Actualizar pip packages
    if [ -f "/home/aegis/openagi/requirements.txt" ]; then
        log_info "Actualizando paquetes Python..."
        cd /home/aegis/openagi
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt --upgrade
        deactivate
    fi
    
    # Reiniciar servicios
    log_info "Reiniciando servicios..."
    supervisorctl restart all
    
    log_success "✅ Actualización completada"
}

# Función de limpieza
perform_cleanup() {
    log_info "Realizando limpieza del sistema..."
    
    # Limpiar paquetes
    apt-get autoremove -y
    apt-get autoclean
    apt-get clean
    
    # Limpiar logs antiguos
    find /var/log -name "*.log.*" -mtime +30 -delete 2>/dev/null || true
    find /var/log -name "*.gz" -mtime +30 -delete 2>/dev/null || true
    
    # Limpiar caché
    sync; echo 3 > /proc/sys/vm/drop_caches
    
    # Limpiar archivos temporales
    find /tmp -type f -atime +7 -delete 2>/dev/null || true
    find /var/tmp -type f -atime +7 -delete 2>/dev/null || true
    
    log_success "✅ Limpieza completada"
}

# Función de envío de alertas
send_alert() {
    local subject="$1"
    local message="$2"
    
    if command -v mail > /dev/null 2>&1; then
        echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
    fi
    
    # También registrar en el log
    log_error "ALERTA: $subject - $message"
}

# Monitoreo completo
full_monitor() {
    log_info "Iniciando monitoreo completo..."
    
    check_services
    check_resources
    check_logs
    check_network
    check_security
    
    log_success "✅ Monitoreo completo finalizado"
}

# Función de ayuda
show_help() {
    echo "AEGIS Framework Maintenance Script"
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  monitor      - Run full system monitoring"
    echo "  services     - Check service status"
    echo "  resources    - Check resource usage"
    echo "  logs         - Check error logs"
    echo "  network      - Check network connectivity"
    echo "  security     - Check security status"
    echo "  backup       - Perform full backup"
    echo "  update       - Update system and packages"
    echo "  cleanup      - Clean up system"
    echo "  help         - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 monitor"
    echo "  $0 backup"
    echo "  $0 update"
}

# Verificar permisos
if [[ $EUID -ne 0 ]]; then
    log_error "Este script debe ejecutarse como root"
    exit 1
fi

# Crear directorios necesarios
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

# Ejecutar comando
if [[ $# -eq 0 ]]; then
    full_monitor
else
    case "$1" in
        monitor)
            full_monitor
            ;;
        services)
            check_services
            ;;
        resources)
            check_resources
            ;;
        logs)
            check_logs
            ;;
        network)
            check_network
            ;;
        security)
            check_security
            ;;
        backup)
            perform_backup
            ;;
        update)
            perform_update
            ;;
        cleanup)
            perform_cleanup
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Comando desconocido: $1"
            show_help
            exit 1
            ;;
    esac
fi