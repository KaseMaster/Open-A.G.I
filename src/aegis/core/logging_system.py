#!/usr/bin/env python3
"""
AEGIS Framework - Sistema de Logs Distribuidos
Centraliza y gestiona logs de todos los componentes del sistema
"""

import asyncio
import json
import logging
import time
import threading
import queue
import os
import gzip
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import socket
import psutil
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiofiles.os
from collections import defaultdict, deque

# Configuración de logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Niveles de logging extendidos"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"

class LogCategory(Enum):
    """Categorías de logs para clasificación"""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    CONSENSUS = "consensus"
    P2P = "p2p"
    CRYPTO = "crypto"
    MONITORING = "monitoring"
    AUDIT = "audit"
    ERROR = "error"

class LogDestination(Enum):
    """Destinos de logs"""
    FILE = "file"
    CONSOLE = "console"
    NETWORK = "network"
    DATABASE = "database"
    SYSLOG = "syslog"
    ELASTICSEARCH = "elasticsearch"

@dataclass
class LogEntry:
    """Entrada de log estructurada"""
    timestamp: float
    level: LogLevel
    category: LogCategory
    component: str
    node_id: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    function_name: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class LogFilter:
    """Filtro para logs"""
    levels: Optional[List[LogLevel]] = None
    categories: Optional[List[LogCategory]] = None
    components: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    keywords: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class LogRotationConfig:
    """Configuración de rotación de logs"""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_files: int = 10
    rotation_interval: int = 86400  # 24 horas
    compress_old_files: bool = True
    retention_days: int = 30

@dataclass
class LogAggregationRule:
    """Regla de agregación de logs"""
    rule_id: str
    name: str
    pattern: str
    time_window: int  # segundos
    threshold: int
    action: str  # "alert", "suppress", "escalate"
    enabled: bool = True

class LogFormatter:
    """Formateador de logs con múltiples formatos"""
    
    def __init__(self):
        self.formatters = {
            "json": self._format_json,
            "structured": self._format_structured,
            "simple": self._format_simple,
            "detailed": self._format_detailed,
            "syslog": self._format_syslog
        }
    
    def format(self, entry: LogEntry, format_type: str = "json") -> str:
        """Formatear entrada de log"""
        formatter = self.formatters.get(format_type, self._format_json)
        return formatter(entry)
    
    def _format_json(self, entry: LogEntry) -> str:
        """Formato JSON estructurado"""
        data = asdict(entry)
        data['timestamp_iso'] = datetime.fromtimestamp(entry.timestamp).isoformat()
        data['level'] = entry.level.value
        data['category'] = entry.category.value
        return json.dumps(data, ensure_ascii=False)
    
    def _format_structured(self, entry: LogEntry) -> str:
        """Formato estructurado legible"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return (f"[{timestamp}] {entry.level.value:>11} | {entry.category.value:>12} | "
                f"{entry.component:>15} | {entry.node_id} | {entry.message}")
    
    def _format_simple(self, entry: LogEntry) -> str:
        """Formato simple"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')
        return f"{timestamp} {entry.level.value} {entry.component}: {entry.message}"
    
    def _format_detailed(self, entry: LogEntry) -> str:
        """Formato detallado con metadatos"""
        base = self._format_structured(entry)
        if entry.metadata:
            metadata_str = json.dumps(entry.metadata, ensure_ascii=False)
            base += f" | META: {metadata_str}"
        if entry.correlation_id:
            base += f" | CORR: {entry.correlation_id}"
        return base
    
    def _format_syslog(self, entry: LogEntry) -> str:
        """Formato compatible con syslog"""
        priority = self._get_syslog_priority(entry.level)
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%b %d %H:%M:%S')
        hostname = socket.gethostname()
        return f"<{priority}>{timestamp} {hostname} {entry.component}: {entry.message}"
    
    def _get_syslog_priority(self, level: LogLevel) -> int:
        """Convertir nivel a prioridad syslog"""
        mapping = {
            LogLevel.TRACE: 7,      # debug
            LogLevel.DEBUG: 7,      # debug
            LogLevel.INFO: 6,       # info
            LogLevel.WARNING: 4,    # warning
            LogLevel.ERROR: 3,      # error
            LogLevel.CRITICAL: 2,   # critical
            LogLevel.SECURITY: 1,   # alert
            LogLevel.AUDIT: 6,      # info
            LogLevel.PERFORMANCE: 6 # info
        }
        return mapping.get(level, 6)

class LogStorage:
    """Sistema de almacenamiento de logs"""
    
    def __init__(self, base_path: str = "logs", rotation_config: LogRotationConfig = None):
        self.base_path = Path(base_path)
        self.rotation_config = rotation_config or LogRotationConfig()
        self.file_handles: Dict[str, Any] = {}
        self.file_sizes: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        # Crear directorio base
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store(self, entry: LogEntry, format_type: str = "json"):
        """Almacenar entrada de log"""
        try:
            # Determinar archivo de destino
            file_path = self._get_log_file_path(entry)
            
            # Formatear entrada
            formatter = LogFormatter()
            formatted_entry = formatter.format(entry, format_type)
            
            # Escribir de forma asíncrona
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                await f.write(formatted_entry + '\n')
            
            # Actualizar tamaño y verificar rotación
            await self._check_rotation(file_path)
            
        except Exception as e:
            logger.error(f"Error almacenando log: {e}")
    
    def _get_log_file_path(self, entry: LogEntry) -> Path:
        """Obtener ruta del archivo de log"""
        date_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d')
        filename = f"{entry.category.value}_{date_str}.log"
        return self.base_path / entry.component / filename
    
    async def _check_rotation(self, file_path: Path):
        """Verificar si es necesario rotar el archivo"""
        try:
            if not file_path.exists():
                return
            
            file_size = await aiofiles.os.path.getsize(file_path)
            
            if file_size > self.rotation_config.max_file_size:
                await self._rotate_file(file_path)
                
        except Exception as e:
            logger.error(f"Error verificando rotación: {e}")
    
    async def _rotate_file(self, file_path: Path):
        """Rotar archivo de log"""
        try:
            # Crear nombre del archivo rotado
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rotated_name = f"{file_path.stem}_{timestamp}.log"
            
            if self.rotation_config.compress_old_files:
                rotated_name += ".gz"
            
            rotated_path = file_path.parent / rotated_name
            
            # Mover y comprimir si es necesario
            if self.rotation_config.compress_old_files:
                await self._compress_file(file_path, rotated_path)
            else:
                await aiofiles.os.rename(file_path, rotated_path)
            
            # Limpiar archivos antiguos
            await self._cleanup_old_files(file_path.parent, file_path.stem)
            
            logger.info(f"Archivo rotado: {file_path} -> {rotated_path}")
            
        except Exception as e:
            logger.error(f"Error rotando archivo: {e}")
    
    async def _compress_file(self, source: Path, destination: Path):
        """Comprimir archivo de log"""
        try:
            async with aiofiles.open(source, 'rb') as f_in:
                content = await f_in.read()
            
            with gzip.open(destination, 'wb') as f_out:
                f_out.write(content)
            
            await aiofiles.os.remove(source)
            
        except Exception as e:
            logger.error(f"Error comprimiendo archivo: {e}")
    
    async def _cleanup_old_files(self, directory: Path, base_name: str):
        """Limpiar archivos antiguos"""
        try:
            if not directory.exists():
                return
            
            # Buscar archivos relacionados
            pattern = f"{base_name}_*"
            files = list(directory.glob(pattern))
            
            # Ordenar por fecha de modificación
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Eliminar archivos excedentes
            for file_path in files[self.rotation_config.max_files:]:
                await aiofiles.os.remove(file_path)
                logger.info(f"Archivo antiguo eliminado: {file_path}")
            
            # Eliminar archivos por retención
            cutoff_time = time.time() - (self.rotation_config.retention_days * 86400)
            for file_path in files:
                if file_path.stat().st_mtime < cutoff_time:
                    await aiofiles.os.remove(file_path)
                    logger.info(f"Archivo expirado eliminado: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error limpiando archivos antiguos: {e}")

class LogAggregator:
    """Agregador de logs para análisis y alertas"""
    
    def __init__(self):
        self.rules: Dict[str, LogAggregationRule] = {}
        self.counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.time_windows: Dict[str, deque] = defaultdict(deque)
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def add_rule(self, rule: LogAggregationRule):
        """Añadir regla de agregación"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Regla de agregación añadida: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Eliminar regla de agregación"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Regla de agregación eliminada: {rule_id}")
    
    def add_callback(self, rule_id: str, callback: Callable):
        """Añadir callback para regla"""
        self.callbacks[rule_id].append(callback)
    
    async def process_entry(self, entry: LogEntry):
        """Procesar entrada contra reglas de agregación"""
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            if self._matches_pattern(entry, rule.pattern):
                await self._update_counters(rule_id, rule, entry)
    
    def _matches_pattern(self, entry: LogEntry, pattern: str) -> bool:
        """Verificar si la entrada coincide con el patrón"""
        # Implementación simple de coincidencia de patrones
        # En producción, usar regex o patrones más sofisticados
        return pattern.lower() in entry.message.lower()
    
    async def _update_counters(self, rule_id: str, rule: LogAggregationRule, entry: LogEntry):
        """Actualizar contadores para la regla"""
        current_time = entry.timestamp
        window_key = f"{rule_id}_{int(current_time // rule.time_window)}"
        
        # Actualizar contador
        self.counters[rule_id][window_key] += 1
        
        # Mantener ventana de tiempo
        self.time_windows[rule_id].append((current_time, window_key))
        
        # Limpiar ventanas antiguas
        cutoff_time = current_time - rule.time_window
        while (self.time_windows[rule_id] and 
               self.time_windows[rule_id][0][0] < cutoff_time):
            old_time, old_key = self.time_windows[rule_id].popleft()
            if old_key in self.counters[rule_id]:
                del self.counters[rule_id][old_key]
        
        # Verificar umbral
        total_count = sum(self.counters[rule_id].values())
        if total_count >= rule.threshold:
            await self._trigger_action(rule, entry, total_count)
    
    async def _trigger_action(self, rule: LogAggregationRule, entry: LogEntry, count: int):
        """Ejecutar acción de la regla"""
        try:
            if rule.action == "alert":
                await self._send_alert(rule, entry, count)
            elif rule.action == "suppress":
                await self._suppress_logs(rule, entry)
            elif rule.action == "escalate":
                await self._escalate_issue(rule, entry, count)
            
            # Ejecutar callbacks
            for callback in self.callbacks[rule.rule_id]:
                try:
                    await callback(rule, entry, count)
                except Exception as e:
                    logger.error(f"Error ejecutando callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error ejecutando acción de regla: {e}")
    
    async def _send_alert(self, rule: LogAggregationRule, entry: LogEntry, count: int):
        """Enviar alerta"""
        alert_message = (f"ALERTA: Regla '{rule.name}' activada. "
                        f"Conteo: {count}, Umbral: {rule.threshold}")
        
        alert_entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.CRITICAL,
            category=LogCategory.AUDIT,
            component="LogAggregator",
            node_id=entry.node_id,
            message=alert_message,
            metadata={
                "rule_id": rule.rule_id,
                "original_entry": asdict(entry),
                "trigger_count": count
            }
        )
        
        logger.critical(alert_message)
    
    async def _suppress_logs(self, rule: LogAggregationRule, entry: LogEntry):
        """Suprimir logs similares"""
        # Implementar lógica de supresión
        logger.info(f"Suprimiendo logs similares para regla: {rule.name}")
    
    async def _escalate_issue(self, rule: LogAggregationRule, entry: LogEntry, count: int):
        """Escalar problema"""
        # Implementar lógica de escalación
        logger.warning(f"Escalando problema para regla: {rule.name}, conteo: {count}")

class DistributedLogger:
    """Logger distribuido principal"""
    
    def __init__(self, node_id: str = None, config: Dict[str, Any] = None):
        self.node_id = node_id or self._generate_node_id()
        self.config = config or {}
        
        # Componentes
        self.storage = LogStorage(
            base_path=self.config.get("log_path", "logs"),
            rotation_config=LogRotationConfig(**self.config.get("rotation", {}))
        )
        self.aggregator = LogAggregator()
        self.formatter = LogFormatter()
        
        # Cola de procesamiento
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.processing_task = None
        self.is_running = False
        
        # Estadísticas
        self.stats = {
            "entries_processed": 0,
            "entries_dropped": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        # Configurar reglas por defecto
        self._setup_default_rules()
    
    def _generate_node_id(self) -> str:
        """Generar ID único del nodo"""
        hostname = socket.gethostname()
        pid = os.getpid()
        timestamp = int(time.time())
        return f"{hostname}_{pid}_{timestamp}"
    
    def _setup_default_rules(self):
        """Configurar reglas de agregación por defecto"""
        # Regla para errores críticos
        critical_rule = LogAggregationRule(
            rule_id="critical_errors",
            name="Errores Críticos",
            pattern="critical|fatal|emergency",
            time_window=300,  # 5 minutos
            threshold=5,
            action="alert"
        )
        self.aggregator.add_rule(critical_rule)
        
        # Regla para intentos de seguridad
        security_rule = LogAggregationRule(
            rule_id="security_attempts",
            name="Intentos de Seguridad",
            pattern="unauthorized|forbidden|attack|intrusion",
            time_window=600,  # 10 minutos
            threshold=10,
            action="escalate"
        )
        self.aggregator.add_rule(security_rule)
    
    async def start(self):
        """Iniciar el sistema de logging"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_logs())
        
        logger.info(f"🚀 Sistema de Logs Distribuidos iniciado - Nodo: {self.node_id}")
    
    async def stop(self):
        """Detener el sistema de logging"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Procesar logs restantes
        while not self.log_queue.empty():
            try:
                entry = self.log_queue.get_nowait()
                await self._process_single_entry(entry)
            except asyncio.QueueEmpty:
                break
        
        logger.info("✅ Sistema de Logs Distribuidos detenido")
    
    async def log(self, level: LogLevel, category: LogCategory, component: str, 
                  message: str, **kwargs):
        """Registrar entrada de log"""
        try:
            entry = LogEntry(
                timestamp=time.time(),
                level=level,
                category=category,
                component=component,
                node_id=self.node_id,
                message=message,
                **kwargs
            )
            
            # Añadir a cola de procesamiento
            try:
                self.log_queue.put_nowait(entry)
            except asyncio.QueueFull:
                self.stats["entries_dropped"] += 1
                # En caso de cola llena, procesar directamente
                await self._process_single_entry(entry)
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error registrando log: {e}")
    
    async def _process_logs(self):
        """Procesar logs de la cola"""
        while self.is_running:
            try:
                # Obtener entrada con timeout
                entry = await asyncio.wait_for(
                    self.log_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_single_entry(entry)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Error procesando logs: {e}")
    
    async def _process_single_entry(self, entry: LogEntry):
        """Procesar una entrada individual"""
        try:
            # Almacenar en disco
            await self.storage.store(entry)
            
            # Procesar agregaciones
            await self.aggregator.process_entry(entry)
            
            # Actualizar estadísticas
            self.stats["entries_processed"] += 1
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error procesando entrada: {e}")
    
    # Métodos de conveniencia para diferentes niveles
    async def trace(self, category: LogCategory, component: str, message: str, **kwargs):
        await self.log(LogLevel.TRACE, category, component, message, **kwargs)
    
    async def debug(self, category: LogCategory, component: str, message: str, **kwargs):
        await self.log(LogLevel.DEBUG, category, component, message, **kwargs)
    
    async def info(self, category: LogCategory, component: str, message: str, **kwargs):
        await self.log(LogLevel.INFO, category, component, message, **kwargs)
    
    async def warning(self, category: LogCategory, component: str, message: str, **kwargs):
        await self.log(LogLevel.WARNING, category, component, message, **kwargs)
    
    async def error(self, category: LogCategory, component: str, message: str, **kwargs):
        await self.log(LogLevel.ERROR, category, component, message, **kwargs)
    
    async def critical(self, category: LogCategory, component: str, message: str, **kwargs):
        await self.log(LogLevel.CRITICAL, category, component, message, **kwargs)
    
    async def security(self, component: str, message: str, **kwargs):
        await self.log(LogLevel.SECURITY, LogCategory.SECURITY, component, message, **kwargs)
    
    async def audit(self, component: str, message: str, **kwargs):
        await self.log(LogLevel.AUDIT, LogCategory.AUDIT, component, message, **kwargs)
    
    async def performance(self, component: str, message: str, **kwargs):
        await self.log(LogLevel.PERFORMANCE, LogCategory.PERFORMANCE, component, message, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        uptime = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "queue_size": self.log_queue.qsize(),
            "is_running": self.is_running,
            "node_id": self.node_id
        }

# Instancia global del logger distribuido
_global_logger: Optional[DistributedLogger] = None

async def initialize_logging(node_id: str = None, config: Dict[str, Any] = None):
    """Inicializar sistema de logging global"""
    global _global_logger
    
    if _global_logger is not None:
        await _global_logger.stop()
    
    _global_logger = DistributedLogger(node_id, config)
    await _global_logger.start()
    
    return _global_logger

def get_logger() -> Optional[DistributedLogger]:
    """Obtener instancia del logger global"""
    return _global_logger

async def shutdown_logging():
    """Cerrar sistema de logging global"""
    global _global_logger
    
    if _global_logger is not None:
        await _global_logger.stop()
        _global_logger = None

# Función principal para demostración
async def main():
    """Función principal de demostración"""
    try:
        logger.info("🎯 Iniciando demostración del Sistema de Logs Distribuidos AEGIS")
        
        # Inicializar sistema
        distributed_logger = await initialize_logging(
            node_id="demo_node",
            config={
                "log_path": "demo_logs",
                "rotation": {
                    "max_file_size": 1024 * 1024,  # 1MB para demo
                    "max_files": 5
                }
            }
        )
        
        # Generar logs de ejemplo
        logger.info("📝 Generando logs de ejemplo...")
        
        await distributed_logger.info(LogCategory.SYSTEM, "DemoComponent", "Sistema iniciado correctamente")
        await distributed_logger.warning(LogCategory.NETWORK, "NetworkManager", "Latencia alta detectada")
        await distributed_logger.error(LogCategory.DATABASE, "DatabaseConnector", "Error de conexión temporal")
        await distributed_logger.security("AuthManager", "Intento de acceso no autorizado detectado")
        await distributed_logger.performance("PerformanceMonitor", "CPU usage: 85%", metadata={"cpu_percent": 85})
        
        # Generar logs para activar reglas de agregación
        for i in range(6):
            await distributed_logger.critical(LogCategory.ERROR, "CriticalSystem", f"Error crítico #{i+1}")
            await asyncio.sleep(0.1)
        
        # Esperar procesamiento
        await asyncio.sleep(2)
        
        # Mostrar estadísticas
        stats = distributed_logger.get_stats()
        logger.info(f"📊 Estadísticas del sistema:")
        logger.info(f"   - Entradas procesadas: {stats['entries_processed']}")
        logger.info(f"   - Entradas descartadas: {stats['entries_dropped']}")
        logger.info(f"   - Errores: {stats['errors']}")
        logger.info(f"   - Tiempo activo: {stats['uptime_seconds']:.2f}s")
        
        # Cerrar sistema
        await shutdown_logging()
        
        logger.info("✅ Demostración completada exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error en demostración: {e}")

if __name__ == "__main__":
    asyncio.run(main())