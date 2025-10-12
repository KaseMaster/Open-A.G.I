# Guía de Resolución de Problemas AEGIS

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Herramientas de Diagnóstico](#herramientas-de-diagnóstico)
3. [Problemas de Inicio y Configuración](#problemas-de-inicio-y-configuración)
4. [Problemas de Red P2P](#problemas-de-red-p2p)
5. [Problemas de Almacenamiento](#problemas-de-almacenamiento)
6. [Problemas de Consenso](#problemas-de-consenso)
7. [Problemas de Rendimiento](#problemas-de-rendimiento)
8. [Problemas de Seguridad](#problemas-de-seguridad)
9. [Problemas de Monitoreo](#problemas-de-monitoreo)
10. [Recuperación de Desastres](#recuperación-de-desastres)
11. [Logs y Análisis](#logs-y-análisis)
12. [Contacto y Soporte](#contacto-y-soporte)

## Introducción

Esta guía proporciona soluciones sistemáticas para los problemas más comunes que pueden surgir al operar AEGIS. Cada sección incluye síntomas, diagnósticos y soluciones paso a paso.

### Metodología de Resolución

1. **Identificar**: Reconocer síntomas y recopilar información
2. **Diagnosticar**: Usar herramientas para determinar la causa raíz
3. **Resolver**: Aplicar soluciones específicas
4. **Verificar**: Confirmar que el problema se ha resuelto
5. **Prevenir**: Implementar medidas para evitar recurrencia

## Herramientas de Diagnóstico

### Script de Diagnóstico Integral

```bash
#!/bin/bash
# scripts/aegis_diagnostics.sh

set -euo pipefail

AEGIS_HOME="${AEGIS_HOME:-/opt/aegis}"
REPORT_DIR="/tmp/aegis_diagnostics"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/aegis_diagnostic_$TIMESTAMP.txt"

# Crear directorio de reportes
mkdir -p "$REPORT_DIR"

# Función de logging
log_section() {
    echo -e "\n## $1" >> "$REPORT_FILE"
    echo "Generated at: $(date)" >> "$REPORT_FILE"
    echo "----------------------------------------" >> "$REPORT_FILE"
}

# Función para ejecutar comandos de forma segura
safe_exec() {
    local cmd="$1"
    local description="$2"
    
    echo "Executing: $description" >&2
    {
        echo "Command: $cmd"
        eval "$cmd" 2>&1 || echo "Command failed with exit code $?"
        echo ""
    } >> "$REPORT_FILE"
}

# Inicializar reporte
echo "AEGIS Comprehensive Diagnostic Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "Host: $(hostname)" >> "$REPORT_FILE"
echo "User: $(whoami)" >> "$REPORT_FILE"
echo "=========================================" >> "$REPORT_FILE"

# 1. Información del Sistema
log_section "System Information"
safe_exec "uname -a" "System kernel information"
safe_exec "cat /etc/os-release" "OS release information"
safe_exec "uptime" "System uptime"
safe_exec "date" "Current date and time"
safe_exec "timedatectl status" "Time synchronization status"

# 2. Recursos del Sistema
log_section "System Resources"
safe_exec "free -h" "Memory usage"
safe_exec "df -h" "Disk usage"
safe_exec "lscpu" "CPU information"
safe_exec "iostat -x 1 3" "I/O statistics"
safe_exec "vmstat 1 3" "Virtual memory statistics"

# 3. Estado del Servicio AEGIS
log_section "AEGIS Service Status"
safe_exec "systemctl status aegis" "Systemd service status"
safe_exec "systemctl is-enabled aegis" "Service enable status"
safe_exec "journalctl -u aegis --no-pager -n 50" "Recent service logs"

# 4. Procesos AEGIS
log_section "AEGIS Processes"
safe_exec "ps aux | grep -E '(aegis|python.*main.py)'" "AEGIS processes"
safe_exec "pgrep -f aegis | xargs -I {} ps -p {} -o pid,ppid,cmd,etime,%cpu,%mem" "Detailed process info"

# 5. Conectividad de Red
log_section "Network Connectivity"
safe_exec "netstat -tlnp | grep -E ':(8000|9090|8080)'" "Listening ports"
safe_exec "ss -tlnp | grep -E ':(8000|9090|8080)'" "Socket statistics"
safe_exec "iptables -L -n" "Firewall rules"
safe_exec "ip route show" "Routing table"

# 6. Configuración AEGIS
log_section "AEGIS Configuration"
if [ -f "$AEGIS_HOME/config/production.yaml" ]; then
    safe_exec "cat $AEGIS_HOME/config/production.yaml" "Production configuration"
fi
if [ -f "$AEGIS_HOME/.env" ]; then
    safe_exec "grep -v -E '(PASSWORD|SECRET|KEY)' $AEGIS_HOME/.env" "Environment variables (sanitized)"
fi

# 7. Logs de AEGIS
log_section "AEGIS Logs"
if [ -f "/var/log/aegis/aegis.log" ]; then
    safe_exec "tail -100 /var/log/aegis/aegis.log" "Recent application logs"
fi
if [ -d "$AEGIS_HOME/logs" ]; then
    safe_exec "ls -la $AEGIS_HOME/logs/" "Log directory contents"
    safe_exec "tail -50 $AEGIS_HOME/logs/*.log" "Recent log entries"
fi

# 8. Base de Datos
log_section "Database Connectivity"
if [ -n "${DATABASE_URL:-}" ]; then
    safe_exec "pg_isready -d '$DATABASE_URL'" "PostgreSQL connectivity"
    safe_exec "psql '$DATABASE_URL' -c 'SELECT version();'" "Database version"
    safe_exec "psql '$DATABASE_URL' -c 'SELECT count(*) FROM pg_stat_activity;'" "Active connections"
fi

# 9. Métricas y Salud
log_section "Health and Metrics"
safe_exec "curl -s -m 5 http://localhost:8080/health" "Health endpoint"
safe_exec "curl -s -m 5 http://localhost:9090/metrics | head -20" "Metrics sample"

# 10. Espacio en Disco y Archivos
log_section "File System"
safe_exec "du -sh $AEGIS_HOME/*" "AEGIS directory sizes"
safe_exec "find $AEGIS_HOME -name '*.log' -exec ls -lh {} \;" "Log file sizes"
safe_exec "lsof | grep aegis" "Open files by AEGIS"

echo "Diagnostic report completed: $REPORT_FILE"
echo "Report size: $(du -h $REPORT_FILE | cut -f1)"
```

### Herramientas de Monitoreo en Tiempo Real

```bash
#!/bin/bash
# scripts/live_monitor.sh

# Monitor en tiempo real de AEGIS
watch -n 2 '
echo "=== AEGIS Live Monitor ==="
echo "Time: $(date)"
echo ""

echo "Service Status:"
systemctl is-active aegis 2>/dev/null || echo "INACTIVE"
echo ""

echo "Resource Usage:"
ps aux | grep -E "(aegis|python.*main.py)" | grep -v grep | awk "{print \"CPU: \" \$3 \"% MEM: \" \$4 \"% CMD: \" \$11}"
echo ""

echo "Network Connections:"
netstat -an | grep -E ":(8000|9090|8080)" | wc -l | awk "{print \"Active connections: \" \$1}"
echo ""

echo "Recent Errors (last 5):"
tail -5 /var/log/aegis/aegis.log | grep -i error || echo "No recent errors"
'
```

### Verificador de Salud Automatizado

```python
#!/usr/bin/env python3
# scripts/health_checker.py

import asyncio
import aiohttp
import psutil
import json
import sys
import time
from datetime import datetime
from pathlib import Path

class AEGISHealthChecker:
    def __init__(self):
        self.checks = []
        self.results = {}
        
    async def check_service_running(self):
        """Verificar si el proceso AEGIS está corriendo"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'aegis' in proc.info['name'].lower() or \
                   any('main.py' in cmd for cmd in proc.info['cmdline']):
                    return {
                        'status': 'healthy',
                        'message': f'Process running (PID: {proc.info["pid"]})',
                        'details': {
                            'pid': proc.info['pid'],
                            'cmdline': ' '.join(proc.info['cmdline'])
                        }
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {
            'status': 'unhealthy',
            'message': 'AEGIS process not found',
            'details': {}
        }
    
    async def check_http_endpoints(self):
        """Verificar endpoints HTTP"""
        endpoints = [
            ('health', 'http://localhost:8080/health'),
            ('metrics', 'http://localhost:9090/metrics')
        ]
        
        results = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for name, url in endpoints:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            results[name] = {
                                'status': 'healthy',
                                'message': f'Endpoint responding (HTTP {response.status})',
                                'response_time': response.headers.get('X-Response-Time', 'N/A')
                            }
                        else:
                            results[name] = {
                                'status': 'unhealthy',
                                'message': f'HTTP {response.status}',
                                'details': {'status_code': response.status}
                            }
                except Exception as e:
                    results[name] = {
                        'status': 'unhealthy',
                        'message': f'Connection failed: {str(e)}',
                        'details': {'error': str(e)}
                    }
        
        return results
    
    async def check_database_connectivity(self):
        """Verificar conectividad de base de datos"""
        try:
            import asyncpg
            import os
            
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return {
                    'status': 'warning',
                    'message': 'DATABASE_URL not configured',
                    'details': {}
                }
            
            conn = await asyncpg.connect(database_url)
            result = await conn.fetchval('SELECT 1')
            await conn.close()
            
            return {
                'status': 'healthy',
                'message': 'Database connection successful',
                'details': {'query_result': result}
            }
            
        except ImportError:
            return {
                'status': 'warning',
                'message': 'asyncpg not available for DB check',
                'details': {}
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Database connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def check_system_resources(self):
        """Verificar recursos del sistema"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memoria
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disco
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Determinar estado general
        status = 'healthy'
        warnings = []
        
        if cpu_percent > 80:
            status = 'warning'
            warnings.append(f'High CPU usage: {cpu_percent}%')
        
        if memory_percent > 85:
            status = 'warning'
            warnings.append(f'High memory usage: {memory_percent}%')
        
        if disk_percent > 90:
            status = 'unhealthy'
            warnings.append(f'High disk usage: {disk_percent}%')
        
        return {
            'status': status,
            'message': '; '.join(warnings) if warnings else 'System resources normal',
            'details': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2)
            }
        }
    
    async def check_log_files(self):
        """Verificar archivos de log"""
        log_paths = [
            '/var/log/aegis/aegis.log',
            '/opt/aegis/logs/aegis.log',
            './logs/aegis.log'
        ]
        
        for log_path in log_paths:
            path = Path(log_path)
            if path.exists():
                # Verificar errores recientes
                try:
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        recent_lines = lines[-100:]  # Últimas 100 líneas
                        
                    error_count = sum(1 for line in recent_lines if 'ERROR' in line.upper())
                    warning_count = sum(1 for line in recent_lines if 'WARNING' in line.upper())
                    
                    if error_count > 10:
                        status = 'unhealthy'
                        message = f'High error count in logs: {error_count} errors'
                    elif error_count > 0 or warning_count > 20:
                        status = 'warning'
                        message = f'Errors/warnings in logs: {error_count} errors, {warning_count} warnings'
                    else:
                        status = 'healthy'
                        message = 'Log files normal'
                    
                    return {
                        'status': status,
                        'message': message,
                        'details': {
                            'log_file': str(path),
                            'error_count': error_count,
                            'warning_count': warning_count,
                            'file_size_mb': round(path.stat().st_size / (1024**2), 2)
                        }
                    }
                except Exception as e:
                    return {
                        'status': 'warning',
                        'message': f'Could not read log file: {str(e)}',
                        'details': {'error': str(e)}
                    }
        
        return {
            'status': 'warning',
            'message': 'No log files found',
            'details': {'searched_paths': log_paths}
        }
    
    async def run_all_checks(self):
        """Ejecutar todas las verificaciones"""
        checks = {
            'service': self.check_service_running(),
            'endpoints': self.check_http_endpoints(),
            'database': self.check_database_connectivity(),
            'resources': self.check_system_resources(),
            'logs': self.check_log_files()
        }
        
        results = {}
        for name, check_coro in checks.items():
            try:
                results[name] = await check_coro
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'message': f'Check failed: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        return results
    
    def generate_report(self, results):
        """Generar reporte de salud"""
        overall_status = 'healthy'
        
        # Determinar estado general
        for check_name, result in results.items():
            if isinstance(result, dict):
                if result.get('status') == 'unhealthy':
                    overall_status = 'unhealthy'
                    break
                elif result.get('status') in ['warning', 'error'] and overall_status == 'healthy':
                    overall_status = 'warning'
            else:
                # Para endpoints que devuelven dict de resultados
                for endpoint_result in result.values():
                    if endpoint_result.get('status') == 'unhealthy':
                        overall_status = 'unhealthy'
                        break
                    elif endpoint_result.get('status') in ['warning', 'error'] and overall_status == 'healthy':
                        overall_status = 'warning'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'checks': results,
            'summary': {
                'total_checks': len(results),
                'healthy': 0,
                'warning': 0,
                'unhealthy': 0,
                'error': 0
            }
        }
        
        # Contar estados
        for check_name, result in results.items():
            if isinstance(result, dict) and 'status' in result:
                status = result['status']
                if status in report['summary']:
                    report['summary'][status] += 1
            else:
                # Para endpoints
                for endpoint_result in result.values():
                    status = endpoint_result.get('status', 'unknown')
                    if status in report['summary']:
                        report['summary'][status] += 1
        
        return report

async def main():
    checker = AEGISHealthChecker()
    results = await checker.run_all_checks()
    report = checker.generate_report(results)
    
    # Imprimir reporte
    print(json.dumps(report, indent=2))
    
    # Exit code basado en estado general
    if report['overall_status'] == 'unhealthy':
        sys.exit(2)
    elif report['overall_status'] == 'warning':
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    asyncio.run(main())
```

## Problemas de Inicio y Configuración

### Problema: AEGIS no inicia

**Síntomas:**
- El servicio falla al iniciar
- Logs muestran errores de configuración
- Proceso termina inmediatamente

**Diagnóstico:**
```bash
# Verificar estado del servicio
systemctl status aegis

# Verificar logs de inicio
journalctl -u aegis -f

# Verificar configuración
python -c "import yaml; yaml.safe_load(open('config/production.yaml'))"

# Verificar dependencias
pip check
```

**Soluciones:**

#### 1. Error de Configuración YAML
```bash
# Validar sintaxis YAML
python -c "
import yaml
try:
    with open('config/production.yaml') as f:
        yaml.safe_load(f)
    print('Configuration is valid')
except yaml.YAMLError as e:
    print(f'YAML Error: {e}')
"

# Corregir formato si es necesario
yamllint config/production.yaml
```

#### 2. Variables de Entorno Faltantes
```bash
# Verificar variables requeridas
cat > scripts/check_env.py << 'EOF'
import os
required_vars = [
    'AEGIS_NODE_ID',
    'DATABASE_URL',
    'AEGIS_ENCRYPTION_KEY'
]

missing = []
for var in required_vars:
    if not os.getenv(var):
        missing.append(var)

if missing:
    print(f"Missing environment variables: {', '.join(missing)}")
    exit(1)
else:
    print("All required environment variables are set")
EOF

python scripts/check_env.py
```

#### 3. Permisos de Archivos
```bash
# Verificar y corregir permisos
sudo chown -R aegis:aegis /opt/aegis
sudo chmod -R 755 /opt/aegis
sudo chmod 600 /opt/aegis/.env
sudo chmod 644 /opt/aegis/config/*.yaml
```

#### 4. Dependencias Faltantes
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Verificar dependencias del sistema
sudo apt update
sudo apt install -y python3-dev build-essential libssl-dev
```

### Problema: Configuración de Base de Datos

**Síntomas:**
- Errores de conexión a PostgreSQL
- Timeouts de base de datos
- Tablas no encontradas

**Diagnóstico:**
```bash
# Verificar conectividad
pg_isready -d "$DATABASE_URL"

# Verificar credenciales
psql "$DATABASE_URL" -c "SELECT version();"

# Verificar tablas
psql "$DATABASE_URL" -c "\dt"
```

**Soluciones:**

#### 1. Inicializar Base de Datos
```bash
# Crear base de datos si no existe
createdb aegis

# Ejecutar migraciones
python scripts/init_db.py

# Verificar esquema
psql "$DATABASE_URL" -c "
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';
"
```

#### 2. Configurar Pool de Conexiones
```yaml
# config/production.yaml
storage:
  type: "postgresql"
  connection_string: "${DATABASE_URL}"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
```

## Problemas de Red P2P

### Problema: No se pueden establecer conexiones P2P

**Síntomas:**
- Métricas muestran 0 peers conectados
- Logs indican "No peers available"
- Nodo aislado de la red

**Diagnóstico:**
```bash
# Verificar puertos abiertos
netstat -tlnp | grep :8000
ss -tlnp | grep :8000

# Verificar conectividad externa
telnet bootstrap-node.aegis.network 8000

# Verificar configuración de firewall
sudo ufw status
sudo iptables -L INPUT -n | grep 8000

# Verificar logs P2P
grep -i "p2p\|peer" /var/log/aegis/aegis.log | tail -20
```

**Soluciones:**

#### 1. Configurar Firewall
```bash
# UFW
sudo ufw allow 8000/tcp
sudo ufw reload

# iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4

# Verificar reglas
sudo ufw status numbered
```

#### 2. Configurar NAT/Port Forwarding
```bash
# Para routers domésticos, configurar port forwarding:
# Router IP: 192.168.1.1
# Internal IP: 192.168.1.100
# External Port: 8000 -> Internal Port: 8000

# Verificar IP pública
curl -s https://ipinfo.io/ip

# Verificar conectividad desde exterior
# (ejecutar desde otra máquina)
telnet YOUR_PUBLIC_IP 8000
```

#### 3. Configurar Bootstrap Nodes
```yaml
# config/production.yaml
p2p:
  port: 8000
  host: "0.0.0.0"
  bootstrap_nodes:
    - "node1.aegis.network:8000"
    - "node2.aegis.network:8000"
    - "node3.aegis.network:8000"
  
  # Configuración de reconexión
  reconnect_interval: 30
  max_reconnect_attempts: 10
  connection_timeout: 15
```

### Problema: Alta latencia en red P2P

**Síntomas:**
- Métricas muestran alta latencia de mensajes
- Timeouts frecuentes
- Sincronización lenta

**Diagnóstico:**
```bash
# Verificar latencia de red
for node in node1.aegis.network node2.aegis.network; do
    echo "Testing $node:"
    ping -c 5 $node
    traceroute $node
done

# Verificar métricas de P2P
curl -s http://localhost:9090/metrics | grep -E "(p2p_latency|p2p_message)"

# Verificar carga de red
iftop -i eth0
nethogs
```

**Soluciones:**

#### 1. Optimizar Configuración de Red
```yaml
# config/production.yaml
p2p:
  # Reducir overhead
  message_compression: true
  batch_size: 100
  flush_interval: 100  # ms
  
  # Optimizar buffers
  send_buffer_size: 65536
  recv_buffer_size: 65536
  
  # Configurar timeouts
  ping_interval: 30
  ping_timeout: 10
```

#### 2. Configurar QoS (Quality of Service)
```bash
# Priorizar tráfico AEGIS
sudo tc qdisc add dev eth0 root handle 1: htb default 30
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 100mbit
sudo tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dport 8000 0xffff flowid 1:10
```

## Problemas de Almacenamiento

### Problema: Corrupción de Datos

**Síntomas:**
- Errores de integridad en logs
- Fallos de validación de datos
- Inconsistencias entre nodos

**Diagnóstico:**
```bash
# Verificar integridad de base de datos
psql "$DATABASE_URL" -c "
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public';
"

# Verificar checksums
psql "$DATABASE_URL" -c "SELECT * FROM pg_stat_database WHERE datname = 'aegis';"

# Verificar logs de errores
grep -i "integrity\|corrupt\|checksum" /var/log/aegis/aegis.log
```

**Soluciones:**

#### 1. Reparar Base de Datos
```bash
# Verificar y reparar índices
psql "$DATABASE_URL" -c "REINDEX DATABASE aegis;"

# Analizar estadísticas
psql "$DATABASE_URL" -c "ANALYZE;"

# Verificar consistencia
psql "$DATABASE_URL" -c "
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    most_common_vals
FROM pg_stats 
WHERE schemaname = 'public' 
AND n_distinct < 0;
"
```

#### 2. Restaurar desde Backup
```bash
# Detener servicio
systemctl stop aegis

# Restaurar base de datos
pg_restore -d "$DATABASE_URL" /backup/aegis_backup_latest.sql

# Verificar integridad después de restauración
psql "$DATABASE_URL" -c "
SELECT COUNT(*) as total_records FROM (
    SELECT 'nodes' as table_name, COUNT(*) as count FROM nodes
    UNION ALL
    SELECT 'blocks' as table_name, COUNT(*) as count FROM blocks
    UNION ALL
    SELECT 'transactions' as table_name, COUNT(*) as count FROM transactions
) as counts;
"

# Reiniciar servicio
systemctl start aegis
```

### Problema: Espacio en Disco Insuficiente

**Síntomas:**
- Errores "No space left on device"
- Fallos de escritura
- Rendimiento degradado

**Diagnóstico:**
```bash
# Verificar uso de disco
df -h
du -sh /opt/aegis/*
du -sh /var/log/aegis/*

# Verificar inodos
df -i

# Identificar archivos grandes
find /opt/aegis -type f -size +100M -exec ls -lh {} \;
```

**Soluciones:**

#### 1. Limpieza de Logs
```bash
# Rotar logs manualmente
sudo logrotate -f /etc/logrotate.d/aegis

# Comprimir logs antiguos
find /var/log/aegis -name "*.log" -mtime +7 -exec gzip {} \;

# Eliminar logs muy antiguos
find /var/log/aegis -name "*.log.gz" -mtime +30 -delete
```

#### 2. Configurar Rotación Automática
```bash
# /etc/logrotate.d/aegis
/var/log/aegis/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 aegis aegis
    postrotate
        systemctl reload aegis
    endscript
}
```

#### 3. Limpiar Datos Temporales
```bash
# Limpiar archivos temporales
find /opt/aegis/tmp -type f -mtime +1 -delete

# Limpiar cache si existe
rm -rf /opt/aegis/cache/*

# Vacuum de base de datos
psql "$DATABASE_URL" -c "VACUUM FULL;"
```

## Problemas de Consenso

### Problema: Fallos de Consenso

**Síntomas:**
- Alertas de "ConsensusFailure"
- Nodos fuera de sincronización
- Transacciones no confirmadas

**Diagnóstico:**
```bash
# Verificar estado de consenso
curl -s http://localhost:9090/metrics | grep consensus

# Comparar altura de bloques entre nodos
for node in node1 node2 node3; do
    echo "Node $node:"
    curl -s http://$node:9090/status | jq .block_height
done

# Verificar logs de consenso
grep -i "consensus\|byzantine\|fork" /var/log/aegis/aegis.log | tail -20
```

**Soluciones:**

#### 1. Resincronizar Nodo
```bash
# Detener nodo problemático
systemctl stop aegis

# Limpiar estado local (CUIDADO: esto eliminará datos locales)
rm -rf /opt/aegis/data/consensus/*

# Reiniciar y resincronizar
systemctl start aegis

# Monitorear progreso de sincronización
watch -n 5 'curl -s http://localhost:9090/status | jq .sync_status'
```

#### 2. Ajustar Configuración de Consenso
```yaml
# config/production.yaml
consensus:
  algorithm: "pbft"  # o "raft"
  
  # Timeouts más conservadores
  timeout: 30
  heartbeat_interval: 5
  election_timeout: 15
  
  # Tolerancia a fallos
  max_byzantine_nodes: 1
  min_validators: 3
  
  # Configuración de red
  message_timeout: 10
  retry_attempts: 3
```

#### 3. Verificar Conectividad entre Validadores
```bash
# Script para verificar conectividad
cat > scripts/check_validator_connectivity.sh << 'EOF'
#!/bin/bash

VALIDATORS=("node1.aegis.network:8000" "node2.aegis.network:8000" "node3.aegis.network:8000")

echo "Checking validator connectivity..."
for validator in "${VALIDATORS[@]}"; do
    echo "Testing $validator:"
    if timeout 5 bash -c "</dev/tcp/${validator%:*}/${validator#*:}"; then
        echo "  ✓ Connected"
    else
        echo "  ✗ Failed to connect"
    fi
done
EOF

chmod +x scripts/check_validator_connectivity.sh
./scripts/check_validator_connectivity.sh
```

## Problemas de Rendimiento

### Problema: Alto Uso de CPU

**Síntomas:**
- CPU constantemente > 80%
- Respuestas lentas
- Timeouts frecuentes

**Diagnóstico:**
```bash
# Verificar uso de CPU por proceso
top -p $(pgrep -f aegis)
htop -p $(pgrep -f aegis)

# Profiling de Python
python -m cProfile -o aegis_profile.prof main.py &
sleep 60
kill %1

# Analizar profile
python -c "
import pstats
p = pstats.Stats('aegis_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Verificar threads
ps -eLf | grep aegis
```

**Soluciones:**

#### 1. Optimizar Configuración
```yaml
# config/production.yaml
performance:
  # Limitar workers
  max_workers: 4
  
  # Optimizar pools
  thread_pool_size: 8
  connection_pool_size: 10
  
  # Configurar cache
  cache:
    enabled: true
    max_size: 1000
    ttl: 300
```

#### 2. Optimizar Código Python
```python
# Ejemplo de optimización en hot paths
import functools

@functools.lru_cache(maxsize=1000)
def expensive_computation(data):
    # Cachear computaciones costosas
    return result

# Usar async/await para I/O
async def optimized_handler():
    # Operaciones I/O no bloqueantes
    result = await async_operation()
    return result
```

#### 3. Configurar Límites de Recursos
```bash
# /etc/systemd/system/aegis.service
[Service]
CPUQuota=80%
MemoryMax=2G
TasksMax=1000

# Aplicar cambios
sudo systemctl daemon-reload
sudo systemctl restart aegis
```

### Problema: Uso Excesivo de Memoria

**Síntomas:**
- Memoria > 90%
- OOM kills en logs
- Swapping excesivo

**Diagnóstico:**
```bash
# Verificar uso de memoria
ps aux | grep aegis
pmap $(pgrep -f aegis)

# Verificar memory leaks
valgrind --tool=memcheck --leak-check=full python main.py

# Verificar swap
swapon -s
vmstat 1 5
```

**Soluciones:**

#### 1. Configurar Garbage Collection
```python
# En main.py o módulo de configuración
import gc
import os

# Configurar GC más agresivo
gc.set_threshold(700, 10, 10)

# Forzar GC periódicamente
def periodic_gc():
    collected = gc.collect()
    print(f"GC collected {collected} objects")

# Configurar variables de entorno
os.environ['PYTHONHASHSEED'] = '0'
os.environ['PYTHONOPTIMIZE'] = '1'
```

#### 2. Optimizar Estructuras de Datos
```python
# Usar __slots__ para clases frecuentes
class OptimizedNode:
    __slots__ = ['id', 'address', 'port', 'last_seen']
    
    def __init__(self, id, address, port):
        self.id = id
        self.address = address
        self.port = port
        self.last_seen = time.time()

# Usar generadores en lugar de listas
def process_large_dataset():
    for item in large_dataset_generator():
        yield process_item(item)
```

#### 3. Configurar Límites de Memoria
```yaml
# config/production.yaml
performance:
  memory:
    max_cache_size: 500  # MB
    gc_threshold: 0.8    # 80% de memoria
    
  # Limitar tamaños de buffer
  buffer_sizes:
    network: 64KB
    storage: 128KB
    consensus: 32KB
```

## Problemas de Seguridad

### Problema: Intentos de Acceso No Autorizado

**Síntomas:**
- Logs muestran intentos de autenticación fallidos
- Conexiones desde IPs sospechosas
- Alertas de seguridad

**Diagnóstico:**
```bash
# Verificar logs de autenticación
grep -i "auth\|login\|unauthorized" /var/log/aegis/aegis.log

# Verificar conexiones activas
netstat -an | grep :8000
ss -tuln | grep :8000

# Verificar intentos de fuerza bruta
grep "failed.*auth" /var/log/aegis/aegis.log | awk '{print $1}' | sort | uniq -c | sort -nr
```

**Soluciones:**

#### 1. Configurar Rate Limiting
```yaml
# config/production.yaml
security:
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
    
    # Bloqueo por IP
    block_duration: 3600  # 1 hora
    max_failures: 5
```

#### 2. Implementar Fail2Ban
```bash
# /etc/fail2ban/jail.local
[aegis]
enabled = true
port = 8000
filter = aegis
logpath = /var/log/aegis/aegis.log
maxretry = 5
bantime = 3600
findtime = 600

# /etc/fail2ban/filter.d/aegis.conf
[Definition]
failregex = .*Authentication failed.*<HOST>.*
            .*Unauthorized access.*<HOST>.*
ignoreregex =
```

#### 3. Configurar Firewall Avanzado
```bash
# Permitir solo IPs conocidas para administración
sudo ufw allow from 192.168.1.0/24 to any port 9090

# Limitar conexiones concurrentes
sudo iptables -A INPUT -p tcp --dport 8000 -m connlimit --connlimit-above 50 -j REJECT

# Protección DDoS básica
sudo iptables -A INPUT -p tcp --dport 8000 -m state --state NEW -m recent --set
sudo iptables -A INPUT -p tcp --dport 8000 -m state --state NEW -m recent --update --seconds 60 --hitcount 20 -j DROP
```

### Problema: Certificados SSL Expirados

**Síntomas:**
- Errores de certificado en logs
- Conexiones TLS fallan
- Alertas de navegador

**Diagnóstico:**
```bash
# Verificar certificados
openssl x509 -in /etc/ssl/certs/aegis.crt -text -noout | grep -A 2 "Validity"

# Verificar configuración TLS
openssl s_client -connect localhost:8000 -servername aegis.yourdomain.com

# Verificar fecha de expiración
echo | openssl s_client -connect aegis.yourdomain.com:443 2>/dev/null | openssl x509 -noout -dates
```

**Soluciones:**

#### 1. Renovar Certificados Let's Encrypt
```bash
# Renovar certificados
sudo certbot renew --dry-run
sudo certbot renew

# Configurar renovación automática
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### 2. Generar Certificados Autofirmados (Desarrollo)
```bash
# Generar clave privada
openssl genrsa -out aegis.key 2048

# Generar certificado
openssl req -new -x509 -key aegis.key -out aegis.crt -days 365 \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=aegis.local"

# Instalar certificados
sudo cp aegis.crt /etc/ssl/certs/
sudo cp aegis.key /etc/ssl/private/
sudo chmod 600 /etc/ssl/private/aegis.key
```

## Problemas de Monitoreo

### Problema: Métricas No Disponibles

**Síntomas:**
- Prometheus no puede scrape métricas
- Grafana muestra "No data"
- Endpoint /metrics no responde

**Diagnóstico:**
```bash
# Verificar endpoint de métricas
curl -v http://localhost:9090/metrics

# Verificar configuración de Prometheus
prometheus --config.file=/etc/prometheus/prometheus.yml --web.config.file=/etc/prometheus/web.yml --dry-run

# Verificar logs de Prometheus
journalctl -u prometheus -f
```

**Soluciones:**

#### 1. Verificar Configuración de Métricas
```yaml
# config/production.yaml
monitoring:
  enabled: true
  metrics_port: 9090
  metrics_path: "/metrics"
  
  # Configurar métricas específicas
  metrics:
    - name: "aegis_node_info"
      type: "info"
      description: "Node information"
    
    - name: "aegis_p2p_connections"
      type: "gauge"
      description: "Number of P2P connections"
```

#### 2. Reiniciar Servicios de Monitoreo
```bash
# Reiniciar Prometheus
sudo systemctl restart prometheus

# Reiniciar Grafana
sudo systemctl restart grafana-server

# Verificar configuración
prometheus --config.file=/etc/prometheus/prometheus.yml --check-config
```

#### 3. Configurar Alertas
```yaml
# /etc/prometheus/alert_rules.yml
groups:
- name: aegis.rules
  rules:
  - alert: AegisDown
    expr: up{job="aegis"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "AEGIS instance is down"
      description: "AEGIS instance {{ $labels.instance }} has been down for more than 1 minute."
```

## Recuperación de Desastres

### Procedimiento de Recuperación Completa

#### 1. Evaluación de Daños
```bash
#!/bin/bash
# scripts/disaster_assessment.sh

echo "=== AEGIS Disaster Recovery Assessment ==="
echo "Timestamp: $(date)"

# Verificar servicios críticos
echo -e "\n1. Service Status:"
systemctl is-active aegis || echo "AEGIS service is DOWN"
systemctl is-active postgresql || echo "PostgreSQL is DOWN"
systemctl is-active nginx || echo "Nginx is DOWN"

# Verificar integridad de datos
echo -e "\n2. Data Integrity:"
if [ -f "/opt/aegis/data/blockchain.db" ]; then
    echo "Blockchain data exists"
else
    echo "WARNING: Blockchain data missing"
fi

# Verificar backups disponibles
echo -e "\n3. Available Backups:"
ls -la /backup/aegis/ | tail -5

# Verificar conectividad de red
echo -e "\n4. Network Connectivity:"
ping -c 1 8.8.8.8 >/dev/null && echo "Internet: OK" || echo "Internet: FAILED"

# Verificar espacio en disco
echo -e "\n5. Disk Space:"
df -h | grep -E "(/$|/opt|/var)"
```

#### 2. Recuperación desde Backup
```bash
#!/bin/bash
# scripts/disaster_recovery.sh

set -euo pipefail

BACKUP_DIR="/backup/aegis"
RECOVERY_LOG="/var/log/aegis/recovery.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$RECOVERY_LOG"
}

# Función de recuperación
recover_from_backup() {
    local backup_file="$1"
    
    log "Starting recovery from backup: $backup_file"
    
    # Detener servicios
    log "Stopping services..."
    systemctl stop aegis
    systemctl stop nginx
    
    # Restaurar base de datos
    log "Restoring database..."
    dropdb aegis || true
    createdb aegis
    pg_restore -d aegis "$backup_file"
    
    # Restaurar archivos de configuración
    log "Restoring configuration files..."
    tar -xzf "${backup_file%.sql}.config.tar.gz" -C /opt/aegis/
    
    # Restaurar datos de aplicación
    if [ -f "${backup_file%.sql}.data.tar.gz" ]; then
        log "Restoring application data..."
        tar -xzf "${backup_file%.sql}.data.tar.gz" -C /opt/aegis/
    fi
    
    # Verificar integridad
    log "Verifying data integrity..."
    psql aegis -c "SELECT COUNT(*) FROM nodes;" || {
        log "ERROR: Database integrity check failed"
        return 1
    }
    
    # Reiniciar servicios
    log "Starting services..."
    systemctl start aegis
    systemctl start nginx
    
    # Verificar funcionamiento
    sleep 10
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        log "Recovery completed successfully"
        return 0
    else
        log "ERROR: Service health check failed"
        return 1
    fi
}

# Buscar backup más reciente
latest_backup=$(ls -t "$BACKUP_DIR"/*.sql | head -1)

if [ -z "$latest_backup" ]; then
    log "ERROR: No backups found in $BACKUP_DIR"
    exit 1
fi

log "Found latest backup: $latest_backup"
recover_from_backup "$latest_backup"
```

#### 3. Recuperación de Red P2P
```bash
#!/bin/bash
# scripts/p2p_recovery.sh

# Limpiar estado P2P corrupto
rm -rf /opt/aegis/data/p2p/*

# Resetear configuración de peers
cat > /opt/aegis/config/recovery_peers.yaml << EOF
p2p:
  bootstrap_nodes:
    - "recovery-node-1.aegis.network:8000"
    - "recovery-node-2.aegis.network:8000"
  
  # Configuración conservadora para recuperación
  connection_timeout: 60
  reconnect_interval: 10
  max_peers: 5
EOF

# Reiniciar con configuración de recuperación
systemctl stop aegis
python main.py --config config/recovery_peers.yaml &

# Monitorear recuperación
sleep 30
peer_count=$(curl -s http://localhost:9090/metrics | grep aegis_p2p_peers | awk '{print $2}')

if [ "$peer_count" -gt 0 ]; then
    echo "P2P recovery successful: $peer_count peers connected"
    # Volver a configuración normal
    systemctl restart aegis
else
    echo "P2P recovery failed: no peers connected"
    exit 1
fi
```

## Logs y Análisis

### Configuración de Logging Estructurado

```python
# logging_config.py
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Añadir información adicional si existe
        if hasattr(record, 'node_id'):
            log_entry['node_id'] = record.node_id
        
        if hasattr(record, 'peer_id'):
            log_entry['peer_id'] = record.peer_id
        
        if hasattr(record, 'transaction_id'):
            log_entry['transaction_id'] = record.transaction_id
        
        return json.dumps(log_entry)

# Configurar logging
def setup_logging():
    logger = logging.getLogger('aegis')
    logger.setLevel(logging.INFO)
    
    # Handler para archivo
    file_handler = logging.FileHandler('/var/log/aegis/aegis.log')
    file_handler.setFormatter(StructuredFormatter())
    
    # Handler para consola (desarrollo)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### Análisis de Logs con Scripts

```bash
#!/bin/bash
# scripts/log_analyzer.sh

LOG_FILE="/var/log/aegis/aegis.log"
ANALYSIS_DIR="/tmp/aegis_analysis"
mkdir -p "$ANALYSIS_DIR"

# Análisis de errores por hora
echo "=== Error Analysis by Hour ===" > "$ANALYSIS_DIR/error_analysis.txt"
grep "ERROR" "$LOG_FILE" | \
    awk '{print $1" "$2}' | \
    cut -d: -f1-2 | \
    sort | uniq -c | \
    sort -nr >> "$ANALYSIS_DIR/error_analysis.txt"

# Top errores más frecuentes
echo -e "\n=== Most Frequent Errors ===" >> "$ANALYSIS_DIR/error_analysis.txt"
grep "ERROR" "$LOG_FILE" | \
    awk -F'"message":' '{print $2}' | \
    awk -F'"' '{print $2}' | \
    sort | uniq -c | \
    sort -nr | head -10 >> "$ANALYSIS_DIR/error_analysis.txt"

# Análisis de rendimiento
echo "=== Performance Analysis ===" > "$ANALYSIS_DIR/performance_analysis.txt"
grep -E "(slow|timeout|latency)" "$LOG_FILE" | \
    tail -20 >> "$ANALYSIS_DIR/performance_analysis.txt"

# Análisis de P2P
echo "=== P2P Connection Analysis ===" > "$ANALYSIS_DIR/p2p_analysis.txt"
grep -E "(peer|connection|p2p)" "$LOG_FILE" | \
    grep -E "(connected|disconnected|failed)" | \
    tail -50 >> "$ANALYSIS_DIR/p2p_analysis.txt"

echo "Log analysis completed. Results in $ANALYSIS_DIR/"
```

### Monitoreo de Logs en Tiempo Real

```bash
#!/bin/bash
# scripts/live_log_monitor.sh

LOG_FILE="/var/log/aegis/aegis.log"

# Colores para output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Función para procesar líneas de log
process_log_line() {
    local line="$1"
    
    if echo "$line" | grep -q "ERROR"; then
        echo -e "${RED}[ERROR]${NC} $line"
    elif echo "$line" | grep -q "WARNING"; then
        echo -e "${YELLOW}[WARNING]${NC} $line"
    elif echo "$line" | grep -q "INFO"; then
        echo -e "${GREEN}[INFO]${NC} $line"
    else
        echo "$line"
    fi
}

# Monitor en tiempo real
echo "Monitoring AEGIS logs in real-time..."
echo "Press Ctrl+C to stop"

tail -f "$LOG_FILE" | while read line; do
    process_log_line "$line"
done
```

## Contacto y Soporte

### Canales de Soporte

- **Documentación Oficial**: https://docs.aegis-project.org
- **GitHub Issues**: https://github.com/aegis-project/aegis/issues
- **Comunidad Discord**: https://discord.gg/aegis-project
- **Stack Overflow**: Tag `aegis-blockchain`
- **Email de Soporte**: support@aegis-project.org

### Información para Reportes de Bugs

Al reportar un problema, incluya:

1. **Información del Sistema**:
   - Versión de AEGIS
   - Sistema operativo y versión
   - Versión de Python
   - Configuración de hardware

2. **Descripción del Problema**:
   - Síntomas observados
   - Pasos para reproducir
   - Comportamiento esperado vs actual

3. **Logs y Diagnósticos**:
   - Logs relevantes (últimas 100 líneas)
   - Salida del script de diagnóstico
   - Métricas de sistema

4. **Configuración**:
   - Archivos de configuración (sin secretos)
   - Variables de entorno relevantes

### Template para Reportes

```markdown
## Descripción del Problema
[Descripción clara y concisa del problema]

## Pasos para Reproducir
1. [Primer paso]
2. [Segundo paso]
3. [Ver error]

## Comportamiento Esperado
[Descripción de lo que debería suceder]

## Comportamiento Actual
[Descripción de lo que realmente sucede]

## Información del Sistema
- AEGIS Version: [versión]
- OS: [sistema operativo]
- Python: [versión]
- Hardware: [especificaciones]

## Logs
```
[Logs relevantes aquí]
```

## Configuración
```yaml
[Configuración relevante aquí]
```

## Información Adicional
[Cualquier información adicional que pueda ser útil]
```

---

*Esta guía de resolución de problemas se actualiza continuamente basada en los problemas reportados por la comunidad. Para contribuir con mejoras, visite nuestro repositorio en GitHub.*