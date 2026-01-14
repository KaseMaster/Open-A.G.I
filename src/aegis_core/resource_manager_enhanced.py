#!/usr/bin/env python3
"""
Gestor de Recursos Mejorado para AEGIS Open AGI
Control de memoria, límites de seguridad y optimización de rendimiento
"""

import os
import sys
import psutil
import threading
import time
import logging
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from logging_config import get_logger
    logger = get_logger("ResourceManager")
except ImportError:
    logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Límites de recursos para el sistema"""
    max_memory_mb: int = 1024
    memory_warning_threshold: float = 80.0  # Porcentaje
    memory_critical_threshold: float = 90.0  # Porcentaje
    max_cpu_percent: float = 80.0
    max_disk_usage_percent: float = 90.0
    max_open_files: int = 1024
    max_processes: int = 100

class ResourceManager:
    """Gestor mejorado de recursos con límites de seguridad"""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = {
            'warning': [],
            'critical': [],
            'recovery': []
        }
        self._setup_resource_limits()
        
    def _setup_resource_limits(self):
        """Configura límites del sistema operativo"""
        try:
            # Límite de memoria virtual
            import resource
            max_memory_bytes = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            
            # Límite de archivos abiertos
            resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits.max_open_files, self.limits.max_open_files))
            
            # Límite de procesos
            resource.setrlimit(resource.RLIMIT_NPROC, (self.limits.max_processes, self.limits.max_processes))
            
            logger.info(f"✅ Límites de recursos configurados: {self.limits.max_memory_mb}MB memoria, {self.limits.max_open_files} archivos, {self.limits.max_processes} procesos")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron configurar límites de recursos: {e}")
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Obtiene uso actual de recursos"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = (memory_mb / self.limits.max_memory_mb) * 100
            
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Información del sistema
            system_memory = psutil.virtual_memory()
            system_disk = psutil.disk_usage('/')
            
            return {
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'system_memory_percent': system_memory.percent,
                'system_disk_percent': system_disk.percent,
                'open_files': len(self.process.open_files()),
                'threads': self.process.num_threads()
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo uso de recursos: {e}")
            return {}
    
    def check_resource_health(self) -> Dict[str, any]:
        """Verifica la salud de los recursos"""
        usage = self.get_resource_usage()
        if not usage:
            return {'status': 'unknown', 'message': 'No se pudieron obtener métricas'}
        
        alerts = []
        status = 'healthy'
        
        # Verificar memoria del proceso
        if usage['memory_percent'] >= self.limits.memory_critical_threshold:
            status = 'critical'
            alerts.append(f"Memoria crítica: {usage['memory_percent']:.1f}%")
        elif usage['memory_percent'] >= self.limits.memory_warning_threshold:
            status = 'warning'
            alerts.append(f"Memoria alta: {usage['memory_percent']:.1f}%")
        
        # Verificar CPU
        if usage['cpu_percent'] >= self.limits.max_cpu_percent:
            if status == 'healthy':
                status = 'warning'
            alerts.append(f"CPU alto: {usage['cpu_percent']:.1f}%")
        
        # Verificar disco
        if usage['system_disk_percent'] >= self.limits.max_disk_usage_percent:
            if status == 'healthy':
                status = 'warning'
            alerts.append(f"Disco lleno: {usage['system_disk_percent']:.1f}%")
        
        return {
            'status': status,
            'usage': usage,
            'alerts': alerts,
            'timestamp': time.time()
        }
    
    def start_monitoring(self, interval: int = 30):
        """Inicia monitoreo continuo de recursos"""
        if self.monitoring:
            logger.warning("⚠️ Monitoreo ya está activo")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        logger.info(f"✅ Monitoreo de recursos iniciado (intervalo: {interval}s)")
    
    def stop_monitoring(self):
        """Detiene el monitoreo de recursos"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Monitoreo de recursos detenido")
    
    def _monitor_loop(self, interval: int):
        """Bucle de monitoreo de recursos"""
        while self.monitoring:
            try:
                health = self.check_resource_health()
                
                if health['status'] == 'critical':
                    logger.critical(f"🚨 Estado crítico de recursos: {health['alerts']}")
                    self._trigger_callbacks('critical', health)
                    self._handle_critical_state(health)
                    
                elif health['status'] == 'warning':
                    logger.warning(f"⚠️ Advertencia de recursos: {health['alerts']}")
                    self._trigger_callbacks('warning', health)
                    
                elif health['status'] == 'healthy':
                    self._trigger_callbacks('recovery', health)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ Error en monitoreo de recursos: {e}")
                time.sleep(interval)
    
    def _handle_critical_state(self, health: Dict):
        """Maneja estados críticos de recursos"""
        try:
            # Intentar liberar memoria
            if 'memory_percent' in health['usage']:
                self._force_garbage_collection()
                
            # Si la memoria sigue crítica, considerar reinicio
            if health['usage'].get('memory_percent', 0) >= self.limits.memory_critical_threshold:
                logger.critical("💀 Memoria crítica persistente - considerando reinicio del servicio")
                # Aquí se podría implementar lógica de reinicio automático
                
        except Exception as e:
            logger.error(f"❌ Error manejando estado crítico: {e}")
    
    def _force_garbage_collection(self):
        """Forza recolección de basura"""
        try:
            import gc
            gc.collect()
            logger.info("🧹 Recolección de basura forzada")
        except Exception as e:
            logger.error(f"❌ Error forzando recolección de basura: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """Agrega callback para eventos de recursos"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"✅ Callback agregado para evento: {event_type}")
        else:
            logger.warning(f"⚠️ Tipo de evento desconocido: {event_type}")
    
    def _trigger_callbacks(self, event_type: str, data: Dict):
        """Ejecuta callbacks para eventos"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"❌ Error ejecutando callback: {e}")
    
    def get_health_report(self) -> Dict[str, any]:
        """Genera reporte completo de salud"""
        health = self.check_resource_health()
        usage = self.get_resource_usage()
        
        return {
            'resource_manager': {
                'status': health['status'],
                'limits': {
                    'max_memory_mb': self.limits.max_memory_mb,
                    'memory_warning_threshold': self.limits.memory_warning_threshold,
                    'memory_critical_threshold': self.limits.memory_critical_threshold
                },
                'usage': usage,
                'alerts': health.get('alerts', []),
                'monitoring_active': self.monitoring,
                'callbacks_registered': {
                    event: len(callbacks) for event, callbacks in self.callbacks.items()
                }
            },
            'timestamp': time.time()
        }

# Instancia global para uso fácil
_resource_manager = None

def get_resource_manager(config: Optional[Dict] = None) -> ResourceManager:
    """Obtiene la instancia global del ResourceManager"""
    global _resource_manager
    if _resource_manager is None:
        limits = None
        if config:
            limits = ResourceLimits(
                max_memory_mb=config.get('max_memory_mb', 1024),
                memory_warning_threshold=config.get('memory_warning_threshold', 80.0),
                memory_critical_threshold=config.get('memory_critical_threshold', 90.0),
                max_cpu_percent=config.get('max_cpu_percent', 80.0),
                max_disk_usage_percent=config.get('max_disk_usage_percent', 90.0),
                max_open_files=config.get('max_open_files', 1024),
                max_processes=config.get('max_processes', 100)
            )
        _resource_manager = ResourceManager(limits)
    return _resource_manager

# Funciones de conveniencia
def start_resource_monitoring(interval: int = 30) -> ResourceManager:
    """Inicia el monitoreo de recursos"""
    manager = get_resource_manager()
    manager.start_monitoring(interval)
    return manager

def get_resource_health() -> Dict[str, any]:
    """Obtiene el estado de salud de los recursos"""
    manager = get_resource_manager()
    return manager.get_health_report()

def check_resources() -> Dict[str, any]:
    """Verifica rápidamente el estado de los recursos"""
    manager = get_resource_manager()
    return manager.check_resource_health()