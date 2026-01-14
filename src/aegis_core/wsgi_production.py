#!/usr/bin/env python3
"""
Configuración WSGI para AEGIS Framework
Servidor de producción con Gunicorn/Uvicorn
"""

import os
import sys
import multiprocessing
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from main import create_app
    from resource_manager_enhanced import get_resource_manager
    from logging_config import get_logger
    
    logger = get_logger("WSGI")
    
    # Crear aplicación Flask
    app = create_app()
    
    # Configurar Resource Manager para producción
    resource_config = {
        'max_memory_mb': 512,
        'memory_warning_threshold': 75.0,
        'memory_critical_threshold': 85.0,
        'max_cpu_percent': 70.0,
        'max_disk_usage_percent': 85.0,
        'max_open_files': 512,
        'max_processes': 50
    }
    
    resource_manager = get_resource_manager(resource_config)
    
    # Callback para advertencias de recursos
    def resource_warning_callback(data):
        logger.warning(f"⚠️ Advertencia de recursos: {data.get('alerts', [])}")
    
    def resource_critical_callback(data):
        logger.critical(f"🚨 Estado crítico de recursos: {data.get('alerts', [])}")
        # Aquí se podría implementar lógica de escalado o reinicio
    
    def resource_recovery_callback(data):
        logger.info(f"✅ Recuperación de recursos detectada")
    
    # Registrar callbacks
    resource_manager.add_callback('warning', resource_warning_callback)
    resource_manager.add_callback('critical', resource_critical_callback)
    resource_manager.add_callback('recovery', resource_recovery_callback)
    
    # Iniciar monitoreo de recursos
    resource_manager.start_monitoring(interval=60)
    
    logger.info("🚀 Aplicación WSGI configurada exitosamente")
    
except Exception as e:
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"❌ Error configurando WSGI: {e}")
    raise

# Configuración de Gunicorndef get_gunicorn_config():
    """Configuración optimizada para Gunicorn"""
    
    # Detectar número de núcleos CPU
    workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Máximo 8 workers
    
    return {
        'bind': '0.0.0.0:8080',
        'workers': workers,
        'worker_class': 'sync',
        'worker_connections': 1000,
        'max_requests': 1000,
        'max_requests_jitter': 50,
        'timeout': 30,
        'keepalive': 2,
        'preload_app': True,
        'user': 'www-data',
        'group': 'www-data',
        'umask': 0o022,
        'tmp_upload_dir': '/tmp',
        'secure_scheme_headers': {
            'X-FORWARDED-PROTOCOL': 'ssl',
            'X-FORWARDED-PROTO': 'https',
            'X-FORWARDED-SSL': 'on'
        },
        'forwarded_allow_ips': '*',
        'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
        'accesslog': 'logs/gunicorn_access.log',
        'errorlog': 'logs/gunicorn_error.log',
        'loglevel': 'info',
        'logger_class': 'gunicorn.glogging.Logger',
        'logconfig': None,
        'syslog_addr': 'unix:///var/run/syslog',
        'syslog': False,
        'syslog_prefix': 'aegis',
        'syslog_facility': 'user',
        'enable_stdio_inheritance': False,
        'statsd_host': None,
        'statsd_prefix': '',
        'proc_name': 'aegis-server',
        'default_proc_name': 'aegis-server',
        'pythonpath': str(project_root),
        'paste': None
    }

if __name__ == "__main__":
    # Si se ejecuta directamente, usar configuración de desarrollo
    app.run(host='0.0.0.0', port=8080, debug=False)