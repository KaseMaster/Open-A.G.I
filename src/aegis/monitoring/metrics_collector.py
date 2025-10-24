#!/usr/bin/env python3
"""
AEGIS - Sistema de Métricas Avanzadas
Recolección, análisis y visualización de métricas del sistema
"""

import asyncio
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Tipos de métricas disponibles"""
    SYSTEM = "system"
    NETWORK = "network"
    PERFORMANCE = "performance"
    SECURITY = "security"
    APPLICATION = "application"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MetricValue:
    """Valor de métrica con metadatos"""
    value: Union[int, float, str, bool]
    timestamp: float
    unit: str = ""
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class MetricSeries:
    """Serie temporal de métricas"""
    name: str
    metric_type: MetricType
    values: List[MetricValue]
    description: str = ""
    retention_hours: int = 24
    
    def add_value(self, value: Union[int, float, str, bool], unit: str = "", tags: Dict[str, str] = None):
        """Añadir nuevo valor a la serie"""
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            unit=unit,
            tags=tags or {}
        )
        self.values.append(metric_value)
        self._cleanup_old_values()
    
    def _cleanup_old_values(self):
        """Limpiar valores antiguos según retención"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        self.values = [v for v in self.values if v.timestamp > cutoff_time]
    
    def get_latest(self) -> Optional[MetricValue]:
        """Obtener último valor"""
        return self.values[-1] if self.values else None
    
    def get_average(self, minutes: int = 5) -> Optional[float]:
        """Obtener promedio de los últimos N minutos"""
        cutoff_time = time.time() - (minutes * 60)
        recent_values = [
            v.value for v in self.values 
            if v.timestamp > cutoff_time and isinstance(v.value, (int, float))
        ]
        return statistics.mean(recent_values) if recent_values else None

@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    value: Any
    threshold: Any
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None

class ThresholdRule:
    """Regla de umbral para alertas"""
    
    def __init__(self, metric_name: str, condition: str, threshold: Union[int, float], 
                 level: AlertLevel, message: str):
        self.metric_name = metric_name
        self.condition = condition  # >, <, >=, <=, ==, !=
        self.threshold = threshold
        self.level = level
        self.message = message
    
    def check(self, value: Union[int, float]) -> bool:
        """Verificar si el valor cumple la condición"""
        if self.condition == ">":
            return value > self.threshold
        elif self.condition == "<":
            return value < self.threshold
        elif self.condition == ">=":
            return value >= self.threshold
        elif self.condition == "<=":
            return value <= self.threshold
        elif self.condition == "==":
            return value == self.threshold
        elif self.condition == "!=":
            return value != self.threshold
        return False

class SystemMetricsCollector:
    """Recolector de métricas del sistema"""
    
    def __init__(self):
        self.enabled = True
    
    async def collect_cpu_metrics(self) -> Dict[str, Any]:
        """Recolectar métricas de CPU"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        return {
            "cpu_usage_percent": cpu_percent,
            "cpu_count": cpu_count,
            "cpu_frequency_mhz": cpu_freq.current if cpu_freq else 0,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    
    async def collect_memory_metrics(self) -> Dict[str, Any]:
        """Recolectar métricas de memoria"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "memory_total_bytes": memory.total,
            "memory_used_bytes": memory.used,
            "memory_available_bytes": memory.available,
            "memory_usage_percent": memory.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "swap_usage_percent": swap.percent
        }
    
    async def collect_disk_metrics(self) -> Dict[str, Any]:
        """Recolectar métricas de disco"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            "disk_total_bytes": disk_usage.total,
            "disk_used_bytes": disk_usage.used,
            "disk_free_bytes": disk_usage.free,
            "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100
        }
        
        if disk_io:
            metrics.update({
                "disk_read_bytes": disk_io.read_bytes,
                "disk_write_bytes": disk_io.write_bytes,
                "disk_read_count": disk_io.read_count,
                "disk_write_count": disk_io.write_count
            })
        
        return metrics
    
    async def collect_network_metrics(self) -> Dict[str, Any]:
        """Recolectar métricas de red"""
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        return {
            "network_bytes_sent": net_io.bytes_sent,
            "network_bytes_recv": net_io.bytes_recv,
            "network_packets_sent": net_io.packets_sent,
            "network_packets_recv": net_io.packets_recv,
            "network_connections_count": net_connections
        }

class ApplicationMetricsCollector:
    """Recolector de métricas de aplicación"""
    
    def __init__(self):
        self.custom_metrics: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, List[float]] = {}
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Incrementar contador"""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.counters[key] = self.counters.get(key, 0) + value
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Registrar tiempo de ejecución"""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(duration)
        
        # Mantener solo los últimos 1000 valores
        if len(self.timers[key]) > 1000:
            self.timers[key] = self.timers[key][-1000:]
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Establecer valor de gauge"""
        key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.custom_metrics[key] = {
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        }
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Recolectar métricas de aplicación"""
        metrics = {}
        
        # Contadores
        for key, value in self.counters.items():
            name = key.split(':')[0]
            metrics[f"counter_{name}"] = value
        
        # Timers (promedios)
        for key, values in self.timers.items():
            name = key.split(':')[0]
            if values:
                metrics[f"timer_{name}_avg"] = statistics.mean(values)
                metrics[f"timer_{name}_min"] = min(values)
                metrics[f"timer_{name}_max"] = max(values)
                metrics[f"timer_{name}_count"] = len(values)
        
        # Gauges
        for key, data in self.custom_metrics.items():
            name = key.split(':')[0]
            metrics[f"gauge_{name}"] = data["value"]
        
        return metrics

class MetricsAggregator:
    """Agregador de métricas"""
    
    def __init__(self):
        self.aggregation_rules: Dict[str, Callable] = {
            "sum": sum,
            "avg": statistics.mean,
            "min": min,
            "max": max,
            "count": len
        }
    
    def aggregate_metrics(self, metrics: List[MetricValue], 
                         aggregation: str, window_minutes: int = 5) -> Optional[float]:
        """Agregar métricas en ventana de tiempo"""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [
            m.value for m in metrics 
            if m.timestamp > cutoff_time and isinstance(m.value, (int, float))
        ]
        
        if not recent_values:
            return None
        
        if aggregation in self.aggregation_rules:
            return self.aggregation_rules[aggregation](recent_values)
        
        return None

class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.rules: List[ThresholdRule] = []
        self.alert_callbacks: List[Callable] = []
    
    def add_rule(self, rule: ThresholdRule):
        """Añadir regla de alerta"""
        self.rules.append(rule)
    
    def add_callback(self, callback: Callable):
        """Añadir callback para alertas"""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metric_name: str, value: Union[int, float]):
        """Verificar alertas para una métrica"""
        for rule in self.rules:
            if rule.metric_name == metric_name and rule.check(value):
                alert_id = f"{metric_name}_{rule.condition}_{rule.threshold}_{int(time.time())}"
                
                # Verificar si ya existe una alerta similar activa
                existing_alert = next(
                    (a for a in self.alerts 
                     if a.metric_name == metric_name and not a.resolved), 
                    None
                )
                
                if not existing_alert:
                    alert = Alert(
                        id=alert_id,
                        level=rule.level,
                        message=rule.message.format(value=value, threshold=rule.threshold),
                        metric_name=metric_name,
                        value=value,
                        threshold=rule.threshold,
                        timestamp=time.time()
                    )
                    
                    self.alerts.append(alert)
                    self._trigger_callbacks(alert)
    
    def resolve_alert(self, alert_id: str):
        """Resolver alerta"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtener alertas activas"""
        return [a for a in self.alerts if not a.resolved]
    
    def _trigger_callbacks(self, alert: Alert):
        """Disparar callbacks de alerta"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error en callback de alerta: {e}")

class MetricsStorage:
    """Almacenamiento de métricas"""
    
    def __init__(self, storage_path: str = "aegis_metrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    async def save_metrics(self, metrics: Dict[str, MetricSeries]):
        """Guardar métricas en disco"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.storage_path / f"metrics_{timestamp}.json"
            
            serializable_metrics = {}
            for name, series in metrics.items():
                serializable_metrics[name] = {
                    "name": series.name,
                    "metric_type": series.metric_type.value,
                    "description": series.description,
                    "retention_hours": series.retention_hours,
                    "values": [asdict(v) for v in series.values]
                }
            
            with open(file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Métricas guardadas en {file_path}")
            
        except Exception as e:
            logger.error(f"Error guardando métricas: {e}")
    
    async def load_metrics(self, file_path: str) -> Dict[str, MetricSeries]:
        """Cargar métricas desde disco"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metrics = {}
            for name, series_data in data.items():
                values = [
                    MetricValue(**v) for v in series_data["values"]
                ]
                
                series = MetricSeries(
                    name=series_data["name"],
                    metric_type=MetricType(series_data["metric_type"]),
                    values=values,
                    description=series_data.get("description", ""),
                    retention_hours=series_data.get("retention_hours", 24)
                )
                
                metrics[name] = series
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error cargando métricas: {e}")
            return {}

class MetricsReporter:
    """Generador de reportes de métricas"""
    
    def __init__(self):
        self.report_templates = {
            "system_health": self._generate_system_health_report,
            "performance": self._generate_performance_report,
            "alerts": self._generate_alerts_report
        }
    
    def generate_report(self, report_type: str, metrics: Dict[str, MetricSeries], 
                       alerts: List[Alert]) -> Dict[str, Any]:
        """Generar reporte"""
        if report_type in self.report_templates:
            return self.report_templates[report_type](metrics, alerts)
        
        return {"error": f"Tipo de reporte desconocido: {report_type}"}
    
    def _generate_system_health_report(self, metrics: Dict[str, MetricSeries], 
                                     alerts: List[Alert]) -> Dict[str, Any]:
        """Generar reporte de salud del sistema"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_health",
            "summary": {},
            "details": {},
            "alerts_count": len([a for a in alerts if not a.resolved])
        }
        
        # CPU
        if "cpu_usage_percent" in metrics:
            cpu_series = metrics["cpu_usage_percent"]
            latest_cpu = cpu_series.get_latest()
            avg_cpu = cpu_series.get_average(15)  # 15 minutos
            
            report["summary"]["cpu"] = {
                "current": latest_cpu.value if latest_cpu else 0,
                "average_15min": avg_cpu or 0,
                "status": "healthy" if (avg_cpu or 0) < 80 else "warning"
            }
        
        # Memoria
        if "memory_usage_percent" in metrics:
            mem_series = metrics["memory_usage_percent"]
            latest_mem = mem_series.get_latest()
            avg_mem = mem_series.get_average(15)
            
            report["summary"]["memory"] = {
                "current": latest_mem.value if latest_mem else 0,
                "average_15min": avg_mem or 0,
                "status": "healthy" if (avg_mem or 0) < 85 else "warning"
            }
        
        # Disco
        if "disk_usage_percent" in metrics:
            disk_series = metrics["disk_usage_percent"]
            latest_disk = disk_series.get_latest()
            
            report["summary"]["disk"] = {
                "current": latest_disk.value if latest_disk else 0,
                "status": "healthy" if (latest_disk.value if latest_disk else 0) < 90 else "critical"
            }
        
        return report
    
    def _generate_performance_report(self, metrics: Dict[str, MetricSeries], 
                                   alerts: List[Alert]) -> Dict[str, Any]:
        """Generar reporte de rendimiento"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance",
            "metrics": {},
            "trends": {},
            "recommendations": []
        }
        
        # Analizar tendencias de CPU
        if "cpu_usage_percent" in metrics:
            cpu_series = metrics["cpu_usage_percent"]
            recent_values = [v.value for v in cpu_series.values[-60:]]  # Últimos 60 valores
            
            if len(recent_values) > 10:
                trend = "increasing" if recent_values[-5:] > recent_values[:5] else "stable"
                report["trends"]["cpu"] = trend
                
                if statistics.mean(recent_values) > 80:
                    report["recommendations"].append(
                        "CPU usage is high. Consider optimizing processes or scaling resources."
                    )
        
        return report
    
    def _generate_alerts_report(self, metrics: Dict[str, MetricSeries], 
                              alerts: List[Alert]) -> Dict[str, Any]:
        """Generar reporte de alertas"""
        active_alerts = [a for a in alerts if not a.resolved]
        resolved_alerts = [a for a in alerts if a.resolved]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "type": "alerts",
            "summary": {
                "total_alerts": len(alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len(resolved_alerts)
            },
            "active_alerts": [asdict(a) for a in active_alerts],
            "alert_levels": {
                level.value: len([a for a in active_alerts if a.level == level])
                for level in AlertLevel
            }
        }

class AEGISMetricsCollector:
    """Colector principal de métricas AEGIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics: Dict[str, MetricSeries] = {}
        self.system_collector = SystemMetricsCollector()
        self.app_collector = ApplicationMetricsCollector()
        self.aggregator = MetricsAggregator()
        self.alert_manager = AlertManager()
        self.storage = MetricsStorage(self.config.get("storage_path", "aegis_metrics"))
        self.reporter = MetricsReporter()
        
        self.collection_interval = self.config.get("collection_interval", 30)  # segundos
        self.running = False
        self.collection_task = None
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configurar reglas de alerta por defecto"""
        default_rules = [
            ThresholdRule("cpu_usage_percent", ">", 90, AlertLevel.CRITICAL, 
                         "CPU usage is critically high: {value}%"),
            ThresholdRule("memory_usage_percent", ">", 95, AlertLevel.CRITICAL,
                         "Memory usage is critically high: {value}%"),
            ThresholdRule("disk_usage_percent", ">", 95, AlertLevel.CRITICAL,
                         "Disk usage is critically high: {value}%"),
            ThresholdRule("cpu_usage_percent", ">", 80, AlertLevel.WARNING,
                         "CPU usage is high: {value}%"),
            ThresholdRule("memory_usage_percent", ">", 85, AlertLevel.WARNING,
                         "Memory usage is high: {value}%"),
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    def add_metric_series(self, name: str, metric_type: MetricType, 
                         description: str = "", retention_hours: int = 24):
        """Añadir nueva serie de métricas"""
        self.metrics[name] = MetricSeries(
            name=name,
            metric_type=metric_type,
            values=[],
            description=description,
            retention_hours=retention_hours
        )
    
    def record_metric(self, name: str, value: Union[int, float, str, bool], 
                     unit: str = "", tags: Dict[str, str] = None):
        """Registrar valor de métrica"""
        if name not in self.metrics:
            # Auto-crear serie si no existe
            self.add_metric_series(name, MetricType.CUSTOM)
        
        self.metrics[name].add_value(value, unit, tags)
        
        # Verificar alertas para valores numéricos
        if isinstance(value, (int, float)):
            self.alert_manager.check_alerts(name, value)
    
    async def collect_all_metrics(self):
        """Recolectar todas las métricas"""
        try:
            # Métricas del sistema
            if self.system_collector.enabled:
                cpu_metrics = await self.system_collector.collect_cpu_metrics()
                memory_metrics = await self.system_collector.collect_memory_metrics()
                disk_metrics = await self.system_collector.collect_disk_metrics()
                network_metrics = await self.system_collector.collect_network_metrics()
                
                all_system_metrics = {**cpu_metrics, **memory_metrics, **disk_metrics, **network_metrics}
                
                for name, value in all_system_metrics.items():
                    self.record_metric(name, value, tags={"source": "system"})
            
            # Métricas de aplicación
            app_metrics = await self.app_collector.collect_application_metrics()
            for name, value in app_metrics.items():
                self.record_metric(name, value, tags={"source": "application"})
            
            logger.debug(f"Recolectadas {len(all_system_metrics) + len(app_metrics)} métricas")
            
        except Exception as e:
            logger.error(f"Error recolectando métricas: {e}")
    
    async def start_collection(self):
        """Iniciar recolección automática de métricas"""
        if self.running:
            logger.warning("La recolección de métricas ya está en ejecución")
            return
        
        self.running = True
        logger.info(f"Iniciando recolección de métricas cada {self.collection_interval} segundos")
        
        async def collection_loop():
            while self.running:
                try:
                    await self.collect_all_metrics()
                    await asyncio.sleep(self.collection_interval)
                except Exception as e:
                    logger.error(f"Error en bucle de recolección: {e}")
                    await asyncio.sleep(5)  # Esperar antes de reintentar
        
        self.collection_task = asyncio.create_task(collection_loop())
    
    async def stop_collection(self):
        """Detener recolección de métricas"""
        if not self.running:
            return
        
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Recolección de métricas detenida")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        summary = {
            "total_series": len(self.metrics),
            "total_data_points": sum(len(series.values) for series in self.metrics.values()),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "collection_running": self.running,
            "last_collection": datetime.now().isoformat()
        }
        
        # Métricas más recientes
        latest_metrics = {}
        for name, series in self.metrics.items():
            latest = series.get_latest()
            if latest:
                latest_metrics[name] = {
                    "value": latest.value,
                    "timestamp": latest.timestamp,
                    "unit": latest.unit
                }
        
        summary["latest_metrics"] = latest_metrics
        return summary
    
    def get_metric_history(self, metric_name: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """Obtener historial de una métrica"""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = time.time() - (minutes * 60)
        series = self.metrics[metric_name]
        
        return [
            {
                "value": v.value,
                "timestamp": v.timestamp,
                "unit": v.unit,
                "tags": v.tags
            }
            for v in series.values
            if v.timestamp > cutoff_time
        ]
    
    async def generate_report(self, report_type: str = "system_health") -> Dict[str, Any]:
        """Generar reporte de métricas"""
        alerts = self.alert_manager.alerts
        return self.reporter.generate_report(report_type, self.metrics, alerts)
    
    async def save_metrics_snapshot(self):
        """Guardar snapshot de métricas"""
        await self.storage.save_metrics(self.metrics)
    
    # Métodos de conveniencia para la aplicación
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Incrementar contador de aplicación"""
        self.app_collector.increment_counter(name, value, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Registrar tiempo de ejecución"""
        self.app_collector.record_timer(name, duration, tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Establecer valor de gauge"""
        self.app_collector.set_gauge(name, value, tags)

# Instancia global del colector
_metrics_collector: Optional[AEGISMetricsCollector] = None

async def start_metrics_collector(config: Dict[str, Any] = None) -> AEGISMetricsCollector:
    """Iniciar el colector de métricas"""
    global _metrics_collector
    
    if _metrics_collector is not None:
        logger.warning("El colector de métricas ya está iniciado")
        return _metrics_collector
    
    try:
        _metrics_collector = AEGISMetricsCollector(config)
        
        # Configurar callback de alertas
        def alert_callback(alert: Alert):
            logger.warning(f"🚨 ALERTA {alert.level.value.upper()}: {alert.message}")
        
        _metrics_collector.alert_manager.add_callback(alert_callback)
        
        # Iniciar recolección
        await _metrics_collector.start_collection()
        
        logger.info("✅ Sistema de métricas AEGIS iniciado correctamente")
        return _metrics_collector
        
    except Exception as e:
        logger.error(f"❌ Error iniciando sistema de métricas: {e}")
        raise

async def stop_metrics_collector():
    """Detener el colector de métricas"""
    global _metrics_collector
    
    if _metrics_collector is None:
        return
    
    try:
        await _metrics_collector.stop_collection()
        await _metrics_collector.save_metrics_snapshot()
        _metrics_collector = None
        logger.info("✅ Sistema de métricas detenido correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error deteniendo sistema de métricas: {e}")

def get_metrics_collector() -> Optional[AEGISMetricsCollector]:
    """Obtener instancia del colector de métricas"""
    return _metrics_collector

# Decorador para medir tiempo de ejecución
def measure_time(metric_name: str, tags: Dict[str, str] = None):
    """Decorador para medir tiempo de ejecución de funciones"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if _metrics_collector:
                    _metrics_collector.record_timer(metric_name, duration, tags)
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if _metrics_collector:
                    _metrics_collector.record_timer(metric_name, duration, tags)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Demostración del sistema
async def main():
    """Demostración del sistema de métricas AEGIS"""
    print("🚀 Iniciando demostración del Sistema de Métricas AEGIS")
    
    # Configuración de ejemplo
    config = {
        "collection_interval": 5,  # 5 segundos para demo
        "storage_path": "demo_metrics"
    }
    
    try:
        # Iniciar colector
        collector = await start_metrics_collector(config)
        
        # Simular algunas métricas de aplicación
        for i in range(10):
            collector.increment_counter("demo_requests", tags={"endpoint": "/api/test"})
            collector.set_gauge("demo_active_users", i * 10)
            collector.record_timer("demo_response_time", 0.1 + (i * 0.01))
            
            print(f"📊 Iteración {i+1}: Métricas registradas")
            await asyncio.sleep(2)
        
        # Generar reporte
        print("\n📋 Generando reporte de salud del sistema...")
        report = await collector.generate_report("system_health")
        print(json.dumps(report, indent=2))
        
        # Mostrar resumen
        print("\n📈 Resumen de métricas:")
        summary = collector.get_metrics_summary()
        print(json.dumps(summary, indent=2))
        
        # Mostrar alertas activas
        active_alerts = collector.alert_manager.get_active_alerts()
        if active_alerts:
            print(f"\n🚨 Alertas activas: {len(active_alerts)}")
            for alert in active_alerts:
                print(f"  - {alert.level.value.upper()}: {alert.message}")
        else:
            print("\n✅ No hay alertas activas")
        
        print("\n⏳ Ejecutando por 30 segundos más...")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo demostración...")
    except Exception as e:
        print(f"❌ Error en demostración: {e}")
    finally:
        await stop_metrics_collector()
        print("✅ Demostración completada")

if __name__ == "__main__":
    asyncio.run(main())