#!/usr/bin/env python3
"""
Configuración del VPS como Servidor Principal - AEGIS Open AGI
Desarrollador: José María Gómez García
Contacto: José María Gómez García@protonmail.com
Versión: 2.0.0
Licencia: MIT
"""

import os
import json
from pathlib import Path

# Configuración del VPS Principal
VPS_CONFIG = {
    "server_info": {
        "ip": "77.237.235.224",
        "hostname": "aegis-main.openagi.network",
        "role": "primary_node",
        "region": "europe",
        "datacenter": "primary"
    },
    
    "network_config": {
        "p2p_port": 8050,
        "dashboard_port": 8051,
        "chat_port": 8086,
        "blockchain_port": 8052,
        "tor_socks_port": 9050,
        "tor_control_port": 9051
    },
    
    "services": {
        "monitoring_dashboard": {
            "enabled": True,
            "port": 8051,
            "host": "0.0.0.0",
            "ssl": True
        },
        "chat_dapp": {
            "enabled": True,
            "port": 8086,
            "host": "0.0.0.0",
            "ssl": True
        },
        "p2p_network": {
            "enabled": True,
            "port": 8050,
            "bootstrap_node": True,
            "max_connections": 100
        },
        "blockchain": {
            "enabled": True,
            "port": 8052,
            "consensus_node": True,
            "validator": True
        },
        "tor_integration": {
            "enabled": True,
            "hidden_service": True,
            "bridge_relay": False
        }
    },
    
    "security": {
        "ssl_cert_path": "/etc/ssl/certs/aegis.crt",
        "ssl_key_path": "/etc/ssl/private/aegis.key",
        "jwt_secret_key": os.urandom(32).hex(),
        "encryption_key": os.urandom(32).hex(),
        "rate_limiting": True,
        "ddos_protection": True
    },
    
    "database": {
        "postgresql": {
            "host": "localhost",
            "port": 5432,
            "database": "openagi_db",
            "user": "openagi",
            "ssl_mode": "require"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "ssl": True
        }
    },
    
    "logging": {
        "level": "INFO",
        "file": "/var/log/openagi/aegis.log",
        "max_size": "100MB",
        "backup_count": 10
    },
    
    "performance": {
        "workers": 4,
        "max_requests": 1000,
        "timeout": 30,
        "keepalive": 2
    }
}

def create_vps_config():
    """Crear archivos de configuración para el VPS"""
    
    # Crear directorio de configuración
    config_dir = Path("/opt/openagi/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar configuración principal
    config_file = config_dir / "vps_config.json"
    with open(config_file, 'w') as f:
        json.dump(VPS_CONFIG, f, indent=2)
    
    print(f"✅ Configuración del VPS guardada en: {config_file}")
    
    # Crear archivo de entorno
    env_file = config_dir / ".env.production"
    env_content = f"""
# Configuración de Producción - VPS Principal
# AEGIS Open AGI OpenAGI

# Información del Servidor
SERVER_IP={VPS_CONFIG['server_info']['ip']}
SERVER_HOSTNAME={VPS_CONFIG['server_info']['hostname']}
SERVER_ROLE={VPS_CONFIG['server_info']['role']}

# Puertos de Servicios
DASHBOARD_PORT={VPS_CONFIG['network_config']['dashboard_port']}
CHAT_PORT={VPS_CONFIG['network_config']['chat_port']}
P2P_PORT={VPS_CONFIG['network_config']['p2p_port']}
BLOCKCHAIN_PORT={VPS_CONFIG['network_config']['blockchain_port']}

# Base de Datos
DB_HOST={VPS_CONFIG['database']['postgresql']['host']}
DB_PORT={VPS_CONFIG['database']['postgresql']['port']}
DB_NAME={VPS_CONFIG['database']['postgresql']['database']}
DB_USER={VPS_CONFIG['database']['postgresql']['user']}

# Redis
REDIS_HOST={VPS_CONFIG['database']['redis']['host']}
REDIS_PORT={VPS_CONFIG['database']['redis']['port']}
REDIS_DB={VPS_CONFIG['database']['redis']['db']}

# Seguridad
JWT_SECRET_KEY={VPS_CONFIG['security']['jwt_secret_key']}
ENCRYPTION_KEY={VPS_CONFIG['security']['encryption_key']}

# Logging
LOG_LEVEL={VPS_CONFIG['logging']['level']}
LOG_FILE={VPS_CONFIG['logging']['file']}

# Modo de Producción
ENVIRONMENT=production
DEBUG=False
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content.strip())
    
    print(f"✅ Archivo de entorno creado en: {env_file}")
    
    return config_file, env_file

if __name__ == "__main__":
    create_vps_config()