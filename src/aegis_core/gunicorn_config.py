#!/usr/bin/env python3
"""
Configuración avanzada de Gunicorn para AEGIS Framework
Optimizada para producción con seguridad y rendimiento mejorados
"""

import os
import multiprocessing
import logging
from pathlib import Path

# Configuración base
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configuración de logging
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
accesslog = str(LOG_DIR / "gunicorn_access.log")
errorlog = str(LOG_DIR / "gunicorn_error.log")

# Configuración de workers
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", 1000))
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 50))
timeout = int(os.getenv("GUNICORN_TIMEOUT", 30))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 2))

# Configuración de red
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
backlog = int(os.getenv("GUNICORN_BACKLOG", 2048))

# Configuración de proceso
daemon = os.getenv("GUNICORN_DAEMON", "false").lower() == "true"
pidfile = os.getenv("GUNICORN_PIDFILE", "/var/run/gunicorn.pid")
user = os.getenv("GUNICORN_USER", "www-data")
group = os.getenv("GUNICORN_GROUP", "www-data")
umask = int(os.getenv("GUNICORN_UMASK", "0o022"), 8)
tmp_upload_dir = os.getenv("GUNICORN_TMP_UPLOAD_DIR", "/tmp")

# Configuración de seguridad
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "*")
proxy_allow_ips = os.getenv("GUNICORN_PROXY_ALLOW_IPS", "*")

# Configuración de rendimiento
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "true").lower() == "true"
sendfile = os.getenv("GUNICORN_SENDFILE", "true").lower() == "true"
reuse_port = os.getenv("GUNICORN_REUSE_PORT", "false").lower() == "true"

# Configuración de logging avanzada
logger_class = "gunicorn.glogging.Logger"
logconfig = None
syslog_addr = os.getenv("GUNICORN_SYSLOG_ADDR", "unix:///var/run/syslog")
syslog = os.getenv("GUNICORN_SYSLOG", "false").lower() == "true"
syslog_prefix = os.getenv("GUNICORN_SYSLOG_PREFIX", "aegis")
syslog_facility = os.getenv("GUNICORN_SYSLOG_FACILITY", "user")
enable_stdio_inheritance = os.getenv("GUNICORN_ENABLE_STDIO_INHERITANCE", "false").lower() == "true"

# Configuración de estadísticas
statsd_host = os.getenv("GUNICORN_STATSD_HOST", None)
statsd_prefix = os.getenv("GUNICORN_STATSD_PREFIX", "")
statsd_tags = os.getenv("GUNICORN_STATSD_TAGS", "")

# Configuración de proceso
proc_name = os.getenv("GUNICORN_PROC_NAME", "aegis-server")
default_proc_name = os.getenv("GUNICORN_DEFAULT_PROC_NAME", "aegis-server")
pythonpath = str(BASE_DIR)
paste = os.getenv("GUNICORN_PASTE", None)
on_starting = None
on_reload = None
when_ready = None
pre_fork = None
post_fork = None
post_worker_init = None
worker_int = None
worker_abort = None
pre_exec = None
pre_request = None
post_request = None
child_exit = None
worker_exit = None
nworkers_changed = None
on_exit = None

# Configuración de SSL (si se habilita)
if os.getenv("GUNICORN_SSL_ENABLED", "false").lower() == "true":
    keyfile = os.getenv("GUNICORN_SSL_KEYFILE", "/etc/ssl/private/aegis.key")
    certfile = os.getenv("GUNICORN_SSL_CERTFILE", "/etc/ssl/certs/aegis.crt")
    ssl_version = int(os.getenv("GUNICORN_SSL_VERSION", "2"))  # TLSv1.2
    cert_reqs = int(os.getenv("GUNICORN_SSL_CERT_REQS", "0"))  # CERT_NONE
    ca_certs = os.getenv("GUNICORN_SSL_CA_CERTS", None)
    suppress_ragged_eofs = os.getenv("GUNICORN_SSL_SUPPRESS_RAGGED_EOF", "true").lower() == "true"
    do_handshake_on_connect = os.getenv("GUNICORN_SSL_DO_HANDSHAKE_ON_CONNECT", "false").lower() == "true"
    ciphers = os.getenv("GUNICORN_SSL_CIPHERS", "TLSv1.2")

# Funciones de callback
def when_ready_callback(server):
    """Callback cuando el servidor está listo"""
    server.log.info("🚀 Servidor Gunicorn listo y escuchando en %s", bind)
    server.log.info("📊 Workers configurados: %d", workers)
    server.log.info("🔒 Modo de seguridad: %s", "SSL" if os.getenv("GUNICORN_SSL_ENABLED") else "HTTP")

def worker_int_callback(worker):
    """Callback cuando un worker recibe SIGINT"""
    worker.log.info("⚠️ Worker %d recibió SIGINT", worker.pid)

def on_exit_callback(server):
    """Callback cuando el servidor se detiene"""
    server.log.info("⏹️ Servidor Gunicorn detenido")

def pre_fork_callback(server, worker):
    """Callback antes de crear un worker"""
    server.log.debug("🔄 Creando worker %d", worker.pid)

def post_fork_callback(server, worker):
    """Callback después de crear un worker"""
    worker.log.debug("✅ Worker %d creado exitosamente", worker.pid)

def child_exit_callback(server, worker):
    """Callback cuando un worker termina"""
    server.log.warning("👋 Worker %d terminado", worker.pid)

# Asignar callbacks
when_ready = when_ready_callback
worker_int = worker_int_callback
on_exit = on_exit_callback
pre_fork = pre_fork_callback
post_fork = post_fork_callback
child_exit = child_exit_callback

# Configuración de monitoreo y salud
def worker_abort_callback(worker):
    """Callback cuando un worker es abortado (timeout)"""
    worker.log.error("⏰ Worker %d abortado por timeout", worker.pid)
    # Aquí se podría implementar lógica de recuperación

def pre_request_callback(worker, req):
    """Callback antes de procesar una petición"""
    worker.log.debug("📨 Petición entrante: %s %s", req.method, req.path)

def post_request_callback(worker, req, environ, resp):
    """Callback después de procesar una petición"""
    worker.log.debug("📤 Petición completada: %d", resp.status_code)

worker_abort = worker_abort_callback
pre_request = pre_request_callback
post_request = post_request_callback

# Configuración específica del worker
def post_worker_init_callback(worker):
    """Callback después de inicializar el worker"""
    worker.log.info("🔧 Worker %d inicializado", worker.pid)
    
    # Inicializar monitoreo de recursos si está disponible
    try:
        from resource_manager_enhanced import get_resource_manager
        
        resource_config = {
            'max_memory_mb': int(os.getenv("WORKER_MAX_MEMORY_MB", 512)),
            'memory_warning_threshold': float(os.getenv("WORKER_MEMORY_WARNING", 75.0)),
            'memory_critical_threshold': float(os.getenv("WORKER_MEMORY_CRITICAL", 85.0)),
            'max_cpu_percent': float(os.getenv("WORKER_MAX_CPU", 70.0)),
            'max_disk_usage_percent': float(os.getenv("WORKER_MAX_DISK", 85.0)),
            'max_open_files': int(os.getenv("WORKER_MAX_FILES", 512)),
            'max_processes': int(os.getenv("WORKER_MAX_PROCESSES", 50))
        }
        
        resource_manager = get_resource_manager(resource_config)
        resource_manager.start_monitoring(interval=30)
        worker.log.info("📊 Monitoreo de recursos iniciado para worker %d", worker.pid)
        
    except ImportError:
        worker.log.debug("⚠️ Gestor de recursos no disponible")
    except Exception as e:
        worker.log.error("❌ Error inicializando monitoreo: %s", e)

post_worker_init = post_worker_init_callback

# Configuración de rendimiento adicional
def nworkers_changed_callback(server, new_value, old_value):
    """Callback cuando cambia el número de workers"""
    server.log.info("👥 Número de workers cambiado: %d -> %d", old_value, new_value)

nworkers_changed = nworkers_changed_callback

# Configuración de depuración (solo en desarrollo)
if os.getenv("GUNICORN_DEBUG", "false").lower() == "true":
    reload = True
    reload_engine = "auto"
    reload_extra_files = []
    check_config = True
    print_config = True
else:
    reload = False
    check_config = False
    print_config = False