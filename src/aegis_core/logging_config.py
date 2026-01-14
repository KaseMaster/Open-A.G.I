#!/usr/bin/env python3
"""
Configuración de logging optimizada para AEGIS Framework
Manejo seguro de Unicode y configuración de producción
"""

import sys
import logging
import io
from typing import Optional
from pathlib import Path

try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False

class UnicodeSafeHandler(logging.Handler):
    """Handler de logging seguro para Unicode"""
    
    def __init__(self, stream: Optional[io.TextIOWrapper] = None):
        super().__init__()
        self.stream = stream or sys.stdout
        self.encoding = self.stream.encoding or 'utf-8'
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Asegurar codificación UTF-8 segura
            if hasattr(self.stream, 'buffer'):
                # Para streams con buffer (archivos, stdout)
                encoded_msg = msg.encode(self.encoding, errors='replace').decode(self.encoding)
                self.stream.write(encoded_msg + '\n')
            else:
                # Para streams sin buffer
                safe_msg = msg.encode(self.encoding, errors='replace').decode(self.encoding)
                self.stream.write(safe_msg + '\n')
            self.stream.flush()
        except Exception:
            self.handleError(record)

class AEGISLogger:
    """Logger personalizado para AEGIS con manejo seguro de Unicode"""
    
    def __init__(self, name: str = "AEGIS"):
        self.name = name
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Configura el logger con manejo seguro de Unicode"""
        if HAS_LOGURU:
            # Configurar loguru con codificación UTF-8 explícita
            loguru_logger.remove()  # Remover configuración por defecto
            loguru_logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO",
                colorize=True,
                backtrace=True,
                diagnose=True
            )
            # Agregar handler para archivo con rotación
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            loguru_logger.add(
                log_dir / "aegis.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="INFO",
                rotation="10 MB",
                retention="10 days",
                compression="gz"
            )
            self.logger = loguru_logger
        else:
            # Fallback: usar logging estándar con manejo Unicode seguro
            self.logger = logging.getLogger(self.name)
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                
                # Handler para consola con codificación segura
                console_handler = UnicodeSafeHandler()
                console_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
                
                # Handler para archivo
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                file_handler = logging.FileHandler(
                    log_dir / "aegis.log",
                    encoding='utf-8',
                    errors='replace'
                )
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log nivel INFO con manejo seguro de Unicode"""
        safe_message = self._safe_unicode(message)
        if self.logger:
            self.logger.info(safe_message, **kwargs)
        else:
            print(f"[INFO] {safe_message}")
    
    def warning(self, message: str, **kwargs):
        """Log nivel WARNING con manejo seguro de Unicode"""
        safe_message = self._safe_unicode(message)
        if self.logger:
            self.logger.warning(safe_message, **kwargs)
        else:
            print(f"[WARNING] {safe_message}")
    
    def error(self, message: str, **kwargs):
        """Log nivel ERROR con manejo seguro de Unicode"""
        safe_message = self._safe_unicode(message)
        if self.logger:
            self.logger.error(safe_message, **kwargs)
        else:
            print(f"[ERROR] {safe_message}")
    
    def success(self, message: str, **kwargs):
        """Log de éxito con manejo seguro de Unicode"""
        safe_message = self._safe_unicode(message)
        if self.logger and HAS_LOGURU:
            self.logger.success(safe_message, **kwargs)
        else:
            print(f"[SUCCESS] {safe_message}")
    
    def debug(self, message: str, **kwargs):
        """Log nivel DEBUG con manejo seguro de Unicode"""
        safe_message = self._safe_unicode(message)
        if self.logger:
            self.logger.debug(safe_message, **kwargs)
        else:
            print(f"[DEBUG] {safe_message}")
    
    def _safe_unicode(self, message: str) -> str:
        """Asegura que el mensaje sea seguro para Unicode"""
        try:
            # Intentar codificar y decodificar para limpiar caracteres problemáticos
            return message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception:
            # Último fallback: eliminar caracteres no ASCII
            return message.encode('ascii', errors='ignore').decode('ascii')

# Logger global para el framework
def get_logger(name: str = "AEGIS") -> AEGISLogger:
    """Obtiene una instancia del logger AEGIS"""
    return AEGISLogger(name)

# Logger por defecto para compatibilidad
logger = get_logger()

# Funciones de compatibilidad hacia atrás
def info(message: str):
    logger.info(message)

def warning(message: str):
    logger.warning(message)

def error(message: str):
    logger.error(message)

def success(message: str):
    logger.success(message)