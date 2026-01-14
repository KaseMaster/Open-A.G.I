#!/usr/bin/env python3
"""
Configuración avanzada de Uvicorn para AEGIS Open AGI
Servidor ASGI de alto rendimiento con soporte para aplicaciones modernas
"""

import os
import ssl
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configuración base
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configuración de logging
LOG_LEVEL = os.getenv("UVICORN_LOG_LEVEL", "info")
ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"
USE_COLORS = os.getenv("UVICORN_USE_COLORS", "true").lower() == "true"

# Configuración del servidor
HOST = os.getenv("UVICORN_HOST", "127.0.0.1")
PORT = int(os.getenv("UVICORN_PORT", 8000))
BIND = f"{HOST}:{PORT}"

# Configuración de workers
WORKERS = int(os.getenv("UVICORN_WORKERS", 1))
LOOP = os.getenv("UVICORN_LOOP", "auto")  # auto, asyncio, uvloop
HTTP = os.getenv("UVICORN_HTTP", "auto")  # auto, h11, httptools
WS = os.getenv("UVICORN_WS", "auto")  # auto, websockets, wsproto
LIFESPAN = os.getenv("UVICORN_LIFESPAN", "auto")  # auto, on, off

# Configuración de rendimiento
INTERFACE = os.getenv("UVICORN_INTERFACE", "auto")  # auto, asgi3, asgi2, wsgi
RELOAD = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
RELOAD_DIRS = os.getenv("UVICORN_RELOAD_DIRS", "").split(",") if os.getenv("UVICORN_RELOAD_DIRS") else []
RELOAD_DELAY = float(os.getenv("UVICORN_RELOAD_DELAY", "0.25"))
RELOAD_INCLUDES = os.getenv("UVICORN_RELOAD_INCLUDES", "").split(",") if os.getenv("UVICORN_RELOAD_INCLUDES") else []
RELOAD_EXCLUDES = os.getenv("UVICORN_RELOAD_EXCLUDES", "").split(",") if os.getenv("UVICORN_RELOAD_EXCLUDES") else []

# Configuración de límites
LIMIT_MAX_REQUESTS = int(os.getenv("UVICORN_LIMIT_MAX_REQUESTS", 1000))
LIMIT_MAX_REQUESTS_JITTER = int(os.getenv("UVICORN_LIMIT_MAX_REQUESTS_JITTER", 50))
TIMEOUT_KEEP_ALIVE = int(os.getenv("UVICORN_TIMEOUT_KEEP_ALIVE", 5))
TIMEOUT_NOTIFY = int(os.getenv("UVICORN_TIMEOUT_NOTIFY", 30))
TIMEOUT_GRACE_PERIOD = int(os.getenv("UVICORN_TIMEOUT_GRACE_PERIOD", 5))

# Configuración de SSL
SSL_KEYFILE = os.getenv("UVICORN_SSL_KEYFILE", "/etc/ssl/private/aegis.key")
SSL_CERTFILE = os.getenv("UVICORN_SSL_CERTFILE", "/etc/ssl/certs/aegis.crt")
SSL_KEYFILE_PASSWORD = os.getenv("UVICORN_SSL_KEYFILE_PASSWORD", None)
SSL_VERSION = int(os.getenv("UVICORN_SSL_VERSION", "17"))  # PROTOCOL_TLS
SSL_CERT_REQS = int(os.getenv("UVICORN_SSL_CERT_REQS", "0"))  # CERT_NONE
SSL_CA_CERTS = os.getenv("UVICORN_SSL_CA_CERTS", None)
SSL_CIPHERS = os.getenv("UVICORN_SSL_CIPHERS", "TLSv1.2")

# Configuración de headers
FORWARDED_ALLOW_IPS = os.getenv("UVICORN_FORWARDED_ALLOW_IPS", "*")
PROXY_HEADERS = os.getenv("UVICORN_PROXY_HEADERS", "true").lower() == "true"
SERVER_HEADER = os.getenv("UVICORN_SERVER_HEADER", "AEGIS-Server")
DATE_HEADER = os.getenv("UVICORN_DATE_HEADER", "true").lower() == "true"

# Configuración de WebSocket
WS_MAX_SIZE = int(os.getenv("UVICORN_WS_MAX_SIZE", 16777216))  # 16MB
WS_PING_INTERVAL = float(os.getenv("UVICORN_WS_PING_INTERVAL", "20.0"))
WS_PING_TIMEOUT = float(os.getenv("UVICORN_WS_PING_TIMEOUT", "20.0"))
WS_PER_MESSAGE_DEFLATE = os.getenv("UVICORN_WS_PER_MESSAGE_DEFLATE", "true").lower() == "true"

# Configuración de HTTP/2
HTTP2_ENABLED = os.getenv("UVICORN_HTTP2_ENABLED", "false").lower() == "true"
HTTP2_MAX_CONCURRENT_STREAMS = int(os.getenv("UVICORN_HTTP2_MAX_CONCURRENT_STREAMS", 100))
HTTP2_MAX_HEADER_LIST_SIZE = int(os.getenv("UVICORN_HTTP2_MAX_HEADER_LIST_SIZE", 16384))

# Configuración de aplicación
APP_MODULE = os.getenv("UVICORN_APP_MODULE", "wsgi_production:app")
FACTORY = os.getenv("UVICORN_FACTORY", "false").lower() == "true"

# Configuración de proceso
DAEMON = os.getenv("UVICORN_DAEMON", "false").lower() == "true"
PID_FILE = os.getenv("UVICORN_PID_FILE", "/var/run/uvicorn.pid")
USER = os.getenv("UVICORN_USER", "www-data")
GROUP = os.getenv("UVICORN_GROUP", "www-data")
UMask = int(os.getenv("UVICORN_UMASK", "0o022"), 8)

# Configuración de headers de seguridad
SECURE_HEADERS = os.getenv("UVICORN_SECURE_HEADERS", "true").lower() == "true"
SECURITY_HEADERS_CONFIG = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

class AEGISUvicornConfig:
    """Configuración avanzada de Uvicorn para AEGIS Open AGI"""
    
    def __init__(self):
        self.setup_logging()
        self.validate_config()
    
    def setup_logging(self):
        """Configurar logging avanzado"""
        log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        
        # Configurar nivel de logging
        numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        
        # Configurar handlers
        handlers = []
        
        # Handler para archivo
        file_handler = logging.FileHandler(LOG_DIR / "uvicorn.log")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # Handler para consola (si no es daemon)
        if not DAEMON:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_formatter = logging.Formatter(log_format)
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        # Configurar logger raíz
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger("uvicorn.config")
        self.logger.info("🚀 Configuración de Uvicorn inicializada")
    
    def validate_config(self):
        """Validar configuración"""
        # Validar workers
        if WORKERS < 1:
            raise ValueError("El número de workers debe ser al menos 1")
        
        # Validar puerto
        if PORT < 1 or PORT > 65535:
            raise ValueError("El puerto debe estar entre 1 y 65535")
        
        # Validar SSL
        if os.getenv("UVICORN_SSL_ENABLED", "false").lower() == "true":
            if not Path(SSL_KEYFILE).exists():
                raise FileNotFoundError(f"Archivo SSL key no encontrado: {SSL_KEYFILE}")
            if not Path(SSL_CERTFILE).exists():
                raise FileNotFoundError(f"Archivo SSL cert no encontrado: {SSL_CERTFILE}")
        
        self.logger.info("✅ Configuración validada exitosamente")
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Obtener contexto SSL configurado"""
        if os.getenv("UVICORN_SSL_ENABLED", "false").lower() != "true":
            return None
        
        context = ssl.SSLContext(ssl_version)
        context.load_cert_chain(SSL_CERTFILE, SSL_KEYFILE, SSL_KEYFILE_PASSWORD)
        
        if SSL_CA_CERTS:
            context.load_verify_locations(SSL_CA_CERTS)
        
        context.verify_mode = SSL_CERT_REQS
        context.set_ciphers(SSL_CIPHERS)
        
        self.logger.info("🔒 Contexto SSL configurado")
        return context
    
    def get_config(self) -> Dict[str, Any]:
        """Obtener configuración completa"""
        config = {
            "host": HOST,
            "port": PORT,
            "uds": None,  # Unix Domain Socket
            "fd": None,   # File Descriptor
            "loop": LOOP,
            "http": HTTP,
            "ws": WS,
            "lifespan": LIFESPAN,
            "interface": INTERFACE,
            "reload": RELOAD,
            "reload_dirs": RELOAD_DIRS,
            "reload_delay": RELOAD_DELAY,
            "reload_includes": RELOAD_INCLUDES,
            "reload_excludes": RELOAD_EXCLUDES,
            "workers": WORKERS,
            "env_file": None,
            "log_config": None,
            "log_level": LOG_LEVEL,
            "access_log": ACCESS_LOG,
            "use_colors": USE_COLORS,
            "proxy_headers": PROXY_HEADERS,
            "server_header": SERVER_HEADER,
            "date_header": DATE_HEADER,
            "forwarded_allow_ips": FORWARDED_ALLOW_IPS,
            "root_path": "",
            "limit_concurrency": None,
            "limit_max_requests": LIMIT_MAX_REQUESTS,
            "limit_max_requests_jitter": LIMIT_MAX_REQUESTS_JITTER,
            "timeout_keep_alive": TIMEOUT_KEEP_ALIVE,
            "timeout_notify": TIMEOUT_NOTIFY,
            "timeout_grace_period": TIMEOUT_GRACE_PERIOD,
            "callback_notify": None,
            "ssl_keyfile": SSL_KEYFILE if os.getenv("UVICORN_SSL_ENABLED", "false").lower() == "true" else None,
            "ssl_certfile": SSL_CERTFILE if os.getenv("UVICORN_SSL_ENABLED", "false").lower() == "true" else None,
            "ssl_keyfile_password": SSL_KEYFILE_PASSWORD,
            "ssl_version": SSL_VERSION,
            "ssl_cert_reqs": SSL_CERT_REQS,
            "ssl_ca_certs": SSL_CA_CERTS,
            "ssl_ciphers": SSL_CIPHERS,
            "headers": list(SECURITY_HEADERS_CONFIG.items()) if SECURE_HEADERS else [],
            "ws_max_size": WS_MAX_SIZE,
            "ws_ping_interval": WS_PING_INTERVAL,
            "ws_ping_timeout": WS_PING_TIMEOUT,
            "ws_per_message_deflate": WS_PER_MESSAGE_DEFLATE,
            "http2_enabled": HTTP2_ENABLED,
            "http2_max_concurrent_streams": HTTP2_MAX_CONCURRENT_STREAMS,
            "http2_max_header_list_size": HTTP2_MAX_HEADER_LIST_SIZE,
            "app": APP_MODULE,
            "factory": FACTORY,
            "daemon": DAEMON,
            "pid": PID_FILE,
            "user": USER,
            "group": GROUP,
            "umask": UMask,
        }
        
        # Filtrar valores None
        config = {k: v for k, v in config.items() if v is not None}
        
        self.logger.info("📋 Configuración de Uvicorn generada")
        return config
    
    def print_config(self):
        """Imprimir configuración actual"""
        self.logger.info("=" * 60)
        self.logger.info("⚙️  CONFIGURACIÓN UVICORN")
        self.logger.info("=" * 60)
        
        config = self.get_config()
        for key, value in config.items():
            if key in ["ssl_keyfile_password"] and value:
                value = "***SECRET***"
            self.logger.info(f"{key:25}: {value}")
        
        self.logger.info("=" * 60)

# Instancia global de configuración
config = AEGISUvicornConfig()

if __name__ == "__main__":
    # Imprimir configuración si se ejecuta directamente
    config.print_config()