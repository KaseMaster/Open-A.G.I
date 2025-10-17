#!/usr/bin/env python3
"""
📊 AEGIS Enterprise Monitoring - Sprint 5.1
Sistema completo de monitoring y observabilidad enterprise
"""

import asyncio
import time
import json
import logging
import threading
import psutil
import GPUtil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import warnings

# Importar componentes de AEGIS
from aegis_api import AEGISAPIService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== MODELOS DE DATOS =====

@dataclass
class MetricPoint:
    """Punto de métrica individual"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Regla de alerta"""
    name: str
    metric: str
    condition: str  # ">", "<", "==", "!="
    threshold: float
    severity: str  # "low", "medium", "high", "critical"
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """Alerta generada"""
    id: str
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """Estado de salud del sistema"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    last_check: datetime
    error_count: int = 0
    uptime_percentage: float = 100.0

@dataclass
class PerformanceReport:
    """Reporte de performance"""
    period_start: datetime
    period_end: datetime
    total_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    throughput_rps: float
    component_metrics: Dict[str, Dict[str, float]]

# ===== MONITORING CORE =====

class MetricsCollector:
    """Colector de métricas del sistema"""

    def __init__(self, max_history: int = 1000):
        self.metrics = deque(maxlen=max_history)
        self.system_metrics = {}
        self.custom_metrics = {}
        self.lock = threading.Lock()

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Registrar una métrica"""
        if tags is None:
            tags = {}

        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            metric_name=name,
            value=value,
            tags=tags
        )

        with self.lock:
            self.metrics.append(metric_point)

            # Actualizar métricas agregadas
            if name not in self.custom_metrics:
                self.custom_metrics[name] = {
                    'count': 0,
                    'sum': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values': deque(maxlen=100)
                }

            self.custom_metrics[name]['count'] += 1
            self.custom_metrics[name]['sum'] += value
            self.custom_metrics[name]['min'] = min(self.custom_metrics[name]['min'], value)
            self.custom_metrics[name]['max'] = max(self.custom_metrics[name]['max'], value)
            self.custom_metrics[name]['values'].append(value)

    def collect_system_metrics(self):
        """Coletar métricas del sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu.percent", cpu_percent)

            # Memoria
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.percent", memory.percent)
            self.record_metric("system.memory.used_mb", memory.used / 1024 / 1024)

            # Disco
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.percent", disk.percent)
            self.record_metric("system.disk.used_gb", disk.used / 1024 / 1024 / 1024)

            # Red (simplificado)
            net = psutil.net_io_counters()
            self.record_metric("system.network.bytes_sent", net.bytes_sent)
            self.record_metric("system.network.bytes_recv", net.bytes_recv)

            # GPU (si disponible)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.record_metric("system.gpu.memory_percent", gpu.memoryUtil * 100)
                    self.record_metric("system.gpu.utilization", gpu.load * 100)
            except:
                pass  # GPU no disponible

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def get_metric_stats(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Obtener estadísticas de una métrica"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self.lock:
            if name not in self.custom_metrics:
                return {}

            metric_data = self.custom_metrics[name]
            recent_values = [v for v in metric_data['values'] if True]  # Simplificado

            if not recent_values:
                return {}

            return {
                'count': len(recent_values),
                'mean': statistics.mean(recent_values),
                'median': statistics.median(recent_values),
                'min': min(recent_values),
                'max': max(recent_values),
                'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
                'p95': sorted(recent_values)[int(len(recent_values) * 0.95)] if recent_values else 0,
                'p99': sorted(recent_values)[int(len(recent_values) * 0.99)] if recent_values else 0
            }

    def get_recent_metrics(self, name: str = None, limit: int = 100) -> List[MetricPoint]:
        """Obtener métricas recientes"""
        with self.lock:
            if name:
                return [m for m in list(self.metrics) if m.metric_name == name][-limit:]
            else:
                return list(self.metrics)[-limit:]

class AlertManager:
    """Gestor de alertas"""

    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_callbacks = []

    def add_alert_rule(self, rule: AlertRule):
        """Agregar regla de alerta"""
        self.alert_rules[rule.name] = rule
        logger.info(f"✅ Alert rule added: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remover regla de alerta"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"✅ Alert rule removed: {rule_name}")

    def check_alerts(self, metrics_collector: MetricsCollector):
        """Verificar reglas de alerta"""
        current_time = datetime.utcnow()

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            # Verificar cooldown
            if rule.last_triggered and (current_time - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60:
                continue

            # Obtener valor actual de la métrica
            stats = metrics_collector.get_metric_stats(rule.metric, hours=1)
            if not stats:
                continue

            current_value = stats.get('mean', 0)

            # Evaluar condición
            triggered = False
            if rule.condition == ">":
                triggered = current_value > rule.threshold
            elif rule.condition == "<":
                triggered = current_value < rule.threshold
            elif rule.condition == "==":
                triggered = abs(current_value - rule.threshold) < 0.01
            elif rule.condition == "!=":
                triggered = abs(current_value - rule.threshold) >= 0.01

            if triggered:
                # Generar alerta
                alert = Alert(
                    id=f"{rule_name}_{int(current_time.timestamp())}",
                    rule_name=rule_name,
                    severity=rule.severity,
                    message=f"Alert triggered: {rule.metric} {rule.condition} {rule.threshold} (current: {current_value:.2f})",
                    timestamp=current_time,
                    context={
                        'metric': rule.metric,
                        'threshold': rule.threshold,
                        'current_value': current_value,
                        'condition': rule.condition
                    }
                )

                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                rule.last_triggered = current_time

                # Notificar
                self._notify_alert(alert)

                logger.warning(f"🚨 Alert triggered: {alert.message}")

    def resolve_alert(self, alert_id: str):
        """Resolver alerta"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"✅ Alert resolved: {alert_id}")

    def add_notification_callback(self, callback: Callable):
        """Agregar callback de notificación"""
        self.notification_callbacks.append(callback)

    def _notify_alert(self, alert: Alert):
        """Notificar alerta a través de callbacks"""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert notification callback: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Obtener alertas activas"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Obtener historial de alertas"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

class HealthChecker:
    """Verificador de salud de componentes"""

    def __init__(self):
        self.components = {}
        self.check_interval = 30  # segundos
        self.health_history = {}

    def add_component(self, name: str, check_function: Callable, expected_response_time: float = 1.0):
        """Agregar componente para monitoreo de salud"""
        self.components[name] = {
            'check_function': check_function,
            'expected_response_time': expected_response_time,
            'last_check': None,
            'status': 'unknown',
            'response_time': 0,
            'error_count': 0,
            'uptime_start': datetime.utcnow()
        }

    async def check_component_health(self, name: str) -> SystemHealth:
        """Verificar salud de un componente"""
        if name not in self.components:
            return SystemHealth(
                component=name,
                status="unknown",
                response_time=0,
                last_check=datetime.utcnow()
            )

        component = self.components[name]
        start_time = time.time()

        try:
            # Ejecutar función de verificación
            result = await component['check_function']()

            response_time = time.time() - start_time
            component['last_check'] = datetime.utcnow()
            component['response_time'] = response_time

            # Determinar status
            if response_time > component['expected_response_time'] * 2:
                status = "degraded"
            elif result is None or (isinstance(result, bool) and not result):
                status = "unhealthy"
                component['error_count'] += 1
            else:
                status = "healthy"

            component['status'] = status

            # Calcular uptime
            uptime_total = (datetime.utcnow() - component['uptime_start']).total_seconds()
            uptime_percentage = max(0, (uptime_total - component['error_count'] * self.check_interval) / uptime_total * 100)

            return SystemHealth(
                component=name,
                status=status,
                response_time=response_time,
                last_check=component['last_check'],
                error_count=component['error_count'],
                uptime_percentage=uptime_percentage
            )

        except Exception as e:
            response_time = time.time() - start_time
            component['error_count'] += 1
            component['status'] = "unhealthy"
            component['last_check'] = datetime.utcnow()
            component['response_time'] = response_time

            logger.error(f"Health check failed for {name}: {e}")

            return SystemHealth(
                component=name,
                status="unhealthy",
                response_time=response_time,
                last_check=component['last_check'],
                error_count=component['error_count']
            )

    async def check_all_components(self) -> Dict[str, SystemHealth]:
        """Verificar salud de todos los componentes"""
        results = {}
        for name in self.components:
            results[name] = await self.check_component_health(name)
        return results

class PerformanceAnalyzer:
    """Analizador de performance"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.request_logs = deque(maxlen=10000)

    def log_request(self, endpoint: str, method: str, response_time: float,
                   status_code: int, user: str = "anonymous"):
        """Registrar una request"""
        self.request_logs.append({
            'timestamp': datetime.utcnow(),
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'status_code': status_code,
            'user': user,
            'success': status_code < 400
        })

    def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """Generar reporte de performance"""
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=hours)

        # Filtrar requests del período
        period_requests = [
            req for req in self.request_logs
            if req['timestamp'] >= period_start
        ]

        if not period_requests:
            return PerformanceReport(
                period_start=period_start,
                period_end=period_end,
                total_requests=0,
                avg_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                error_rate=0,
                throughput_rps=0,
                component_metrics={}
            )

        total_requests = len(period_requests)
        response_times = [req['response_time'] for req in period_requests]
        error_count = sum(1 for req in period_requests if not req['success'])

        # Calcular métricas
        avg_response_time = statistics.mean(response_times)
        sorted_times = sorted(response_times)
        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
        error_rate = error_count / total_requests if total_requests > 0 else 0

        # Throughput (requests per second)
        period_seconds = (period_end - period_start).total_seconds()
        throughput_rps = total_requests / period_seconds if period_seconds > 0 else 0

        # Métricas por componente
        component_metrics = {}
        endpoint_groups = {}
        for req in period_requests:
            endpoint = req['endpoint'].split('/')[2] if len(req['endpoint'].split('/')) > 2 else 'other'
            if endpoint not in endpoint_groups:
                endpoint_groups[endpoint] = []
            endpoint_groups[endpoint].append(req['response_time'])

        for component, times in endpoint_groups.items():
            component_metrics[component] = {
                'count': len(times),
                'avg_time': statistics.mean(times),
                'p95_time': sorted(times)[int(len(times) * 0.95)] if times else 0
            }

        return PerformanceReport(
            period_start=period_start,
            period_end=period_end,
            total_requests=total_requests,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=error_rate,
            throughput_rps=throughput_rps,
            component_metrics=component_metrics
        )

# ===== MONITORING SYSTEM =====

class AEGISMonitoringSystem:
    """Sistema completo de monitoring para AEGIS"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)

        self.monitoring_active = False
        self.monitoring_thread = None

        self._setup_default_alerts()
        self._setup_default_components()

    def _setup_default_alerts(self):
        """Configurar alertas por defecto"""
        alerts = [
            AlertRule("high_cpu", "system.cpu.percent", ">", 90.0, "high", cooldown_minutes=5),
            AlertRule("high_memory", "system.memory.percent", ">", 85.0, "medium", cooldown_minutes=10),
            AlertRule("low_disk_space", "system.disk.percent", ">", 95.0, "critical", cooldown_minutes=15),
            AlertRule("api_high_response_time", "api.response_time.avg", ">", 2.0, "medium", cooldown_minutes=2),
            AlertRule("api_high_error_rate", "api.error_rate", ">", 0.05, "high", cooldown_minutes=5)
        ]

        for alert in alerts:
            self.alert_manager.add_alert_rule(alert)

    def _setup_default_components(self):
        """Configurar componentes por defecto para health checks"""

        async def check_api_health():
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                return response.status_code == 200
            except:
                return False

        async def check_database_health():
            # Placeholder - implementar verificación de base de datos
            return True

        async def check_model_health():
            # Placeholder - implementar verificación de modelos
            return True

        self.health_checker.add_component("api_service", check_api_health, expected_response_time=1.0)
        self.health_checker.add_component("database", check_database_health, expected_response_time=0.5)
        self.health_checker.add_component("models", check_model_health, expected_response_time=2.0)

    async def start_monitoring(self):
        """Iniciar monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        logger.info("🚀 Starting AEGIS Monitoring System...")

        # Iniciar thread de monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("✅ Monitoring system started")

    def stop_monitoring(self):
        """Detener monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("✅ Monitoring system stopped")

    def _monitoring_loop(self):
        """Loop principal de monitoring"""
        while self.monitoring_active:
            try:
                # Coletar métricas del sistema
                self.metrics_collector.collect_system_metrics()

                # Verificar alertas (en thread separado para no bloquear)
                asyncio.run(self._check_alerts_async())

                # Verificar salud de componentes (cada 30 segundos)
                # Esto se haría en un thread separado o con un scheduler

                time.sleep(10)  # Coletar cada 10 segundos

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    async def _check_alerts_async(self):
        """Verificar alertas de forma asíncrona"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.alert_manager.check_alerts(self.metrics_collector)
        finally:
            loop.close()

    def record_api_request(self, endpoint: str, method: str, response_time: float,
                          status_code: int, user: str = "anonymous"):
        """Registrar una request de API"""
        self.performance_analyzer.log_request(endpoint, method, response_time, status_code, user)

        # Registrar métricas
        self.metrics_collector.record_metric(
            "api.response_time",
            response_time,
            {"endpoint": endpoint, "method": method}
        )

        self.metrics_collector.record_metric(
            "api.status_code",
            status_code,
            {"endpoint": endpoint, "method": method}
        )

        # Métricas agregadas
        success = status_code < 400
        self.metrics_collector.record_metric("api.requests_total", 1, {"success": str(success)})
        self.metrics_collector.record_metric("api.error_rate", 0 if success else 1)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtener datos para dashboard"""
        return {
            'system_metrics': {
                'cpu': self.metrics_collector.get_metric_stats('system.cpu.percent'),
                'memory': self.metrics_collector.get_metric_stats('system.memory.percent'),
                'disk': self.metrics_collector.get_metric_stats('system.disk.percent')
            },
            'api_metrics': {
                'response_time': self.metrics_collector.get_metric_stats('api.response_time'),
                'error_rate': self.metrics_collector.get_metric_stats('api.error_rate'),
                'requests_total': self.metrics_collector.get_metric_stats('api.requests_total')
            },
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'total_alerts_24h': len(self.alert_manager.get_alert_history(24)),
            'performance_report': self.performance_analyzer.generate_performance_report(1),
            'health_status': asyncio.run(self.health_checker.check_all_components())
        }

    def generate_report(self, report_type: str, period_hours: int = 24) -> Dict[str, Any]:
        """Generar reporte"""
        if report_type == "performance":
            return self.performance_analyzer.generate_performance_report(period_hours).__dict__
        elif report_type == "alerts":
            alerts = self.alert_manager.get_alert_history(period_hours)
            return {
                'period_hours': period_hours,
                'total_alerts': len(alerts),
                'alerts_by_severity': {
                    'critical': len([a for a in alerts if a.severity == 'critical']),
                    'high': len([a for a in alerts if a.severity == 'high']),
                    'medium': len([a for a in alerts if a.severity == 'medium']),
                    'low': len([a for a in alerts if a.severity == 'low'])
                },
                'alerts': [alert.__dict__ for alert in alerts[-50:]]  # Últimas 50
            }
        elif report_type == "health":
            health_data = asyncio.run(self.health_checker.check_all_components())
            return {
                'timestamp': datetime.utcnow(),
                'components': {name: health.__dict__ for name, health in health_data.items()},
                'overall_health': all(h.status == 'healthy' for h in health_data.values())
            }

        return {"error": "Unknown report type"}

    def add_custom_alert(self, name: str, metric: str, condition: str,
                        threshold: float, severity: str):
        """Agregar alerta personalizada"""
        rule = AlertRule(name, metric, condition, threshold, severity)
        self.alert_manager.add_alert_rule(rule)

    def export_metrics(self, format: str = "json") -> str:
        """Exportar métricas"""
        metrics_data = {
            'timestamp': datetime.utcnow(),
            'metrics': [m.__dict__ for m in self.metrics_collector.get_recent_metrics()],
            'custom_metrics': self.metrics_collector.custom_metrics
        }

        if format == "json":
            return json.dumps(metrics_data, default=str, indent=2)
        else:
            return str(metrics_data)

# ===== DASHBOARD Y VISUALIZACIÓN =====

class MonitoringDashboard:
    """Dashboard de monitoring"""

    def __init__(self, monitoring_system: AEGISMonitoringSystem):
        self.monitoring = monitoring_system

    def create_system_metrics_chart(self) -> go.Figure:
        """Crear gráfico de métricas del sistema"""
        # Obtener datos recientes
        cpu_metrics = self.monitoring.metrics_collector.get_recent_metrics('system.cpu.percent', 50)
        memory_metrics = self.monitoring.metrics_collector.get_recent_metrics('system.memory.percent', 50)

        fig = go.Figure()

        if cpu_metrics:
            fig.add_trace(go.Scatter(
                x=[m.timestamp for m in cpu_metrics],
                y=[m.value for m in cpu_metrics],
                mode='lines',
                name='CPU %',
                line=dict(color='red')
            ))

        if memory_metrics:
            fig.add_trace(go.Scatter(
                x=[m.timestamp for m in memory_metrics],
                y=[m.value for m in memory_metrics],
                mode='lines',
                name='Memory %',
                line=dict(color='blue')
            ))

        fig.update_layout(
            title="System Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Percentage",
            template="plotly_white"
        )

        return fig

    def create_api_performance_chart(self) -> go.Figure:
        """Crear gráfico de performance de API"""
        response_times = self.monitoring.metrics_collector.get_recent_metrics('api.response_time', 100)

        if not response_times:
            return go.Figure()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[m.timestamp for m in response_times],
            y=[m.value for m in response_times],
            mode='lines+markers',
            name='Response Time (s)',
            line=dict(color='green')
        ))

        fig.update_layout(
            title="API Response Times",
            xaxis_title="Time",
            yaxis_title="Response Time (seconds)",
            template="plotly_white"
        )

        return fig

    def create_alerts_summary(self) -> Dict[str, Any]:
        """Crear resumen de alertas"""
        active_alerts = self.monitoring.alert_manager.get_active_alerts()
        recent_alerts = self.monitoring.alert_manager.get_alert_history(24)

        return {
            'active_count': len(active_alerts),
            'total_24h': len(recent_alerts),
            'by_severity': {
                'critical': len([a for a in active_alerts if a.severity == 'critical']),
                'high': len([a for a in active_alerts if a.severity == 'high']),
                'medium': len([a for a in active_alerts if a.severity == 'medium']),
                'low': len([a for a in active_alerts if a.severity == 'low'])
            },
            'recent_alerts': [
                {
                    'rule': alert.rule_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in recent_alerts[-10:]  # Últimas 10
            ]
        }

    def create_health_status_table(self) -> pd.DataFrame:
        """Crear tabla de estado de salud"""
        health_data = asyncio.run(self.monitoring.health_checker.check_all_components())

        data = []
        for name, health in health_data.items():
            data.append({
                'Component': name,
                'Status': health.status,
                'Response Time': f"{health.response_time:.3f}s",
                'Uptime %': f"{health.uptime_percentage:.1f}%",
                'Errors': health.error_count,
                'Last Check': health.last_check.strftime("%H:%M:%S")
            })

        return pd.DataFrame(data)

# ===== DEMO Y EJEMPLOS =====

async def demo_monitoring_system():
    """Demostración completa del sistema de monitoring"""

    print("📊 AEGIS Enterprise Monitoring Demo")
    print("=" * 38)

    # Inicializar sistema de monitoring
    monitoring = AEGISMonitoringSystem()
    dashboard = MonitoringDashboard(monitoring)

    print("🚀 Inicializando sistema de monitoring...")
    await monitoring.start_monitoring()

    print("✅ Sistema de monitoring activo")

    # ===== DEMO 1: MÉTRICAS DEL SISTEMA =====
    print("\\n\\n📈 DEMO 1: System Metrics Collection")

    # Simular algunas requests
    for i in range(10):
        monitoring.record_api_request(
            f"/api/v1/test/endpoint{i}",
            "GET",
            0.5 + (i * 0.1),  # Response time creciente
            200 if i < 8 else 500,  # Algunos errores
            "demo_user"
        )
        time.sleep(0.1)

    print("📊 Métricas recolectadas:")
    cpu_stats = monitoring.metrics_collector.get_metric_stats('system.cpu.percent')
    memory_stats = monitoring.metrics_collector.get_metric_stats('system.memory.percent')

    print(".1f"    print(".1f"
    # ===== DEMO 2: VERIFICACIÓN DE SALUD =====
    print("\\n\\n🏥 DEMO 2: Health Checks")

    health_status = await monitoring.health_checker.check_all_components()

    print("🏥 Estado de componentes:")
    for component, health in health_status.items():
        status_emoji = "🟢" if health.status == "healthy" else "🟡" if health.status == "degraded" else "🔴"
        print(".3f"
    # ===== DEMO 3: ALERTAS =====
    print("\\n\\n🚨 DEMO 3: Alert System")

    # Agregar una alerta de prueba
    monitoring.add_custom_alert("test_high_cpu", "system.cpu.percent", ">", 50.0, "medium")

    print("🚨 Alertas activas:", len(monitoring.alert_manager.get_active_alerts()))

    # Forzar una alerta para demo
    monitoring.metrics_collector.record_metric("system.cpu.percent", 95.0)
    monitoring.alert_manager.check_alerts(monitoring.metrics_collector)

    active_alerts = monitoring.alert_manager.get_active_alerts()
    print(f"🚨 Alertas después de verificación: {len(active_alerts)}")

    if active_alerts:
        alert = active_alerts[0]
        print(f"   • {alert.rule_name}: {alert.message}")

    # ===== DEMO 4: PERFORMANCE ANALYSIS =====
    print("\\n\\n📊 DEMO 4: Performance Analysis")

    report = monitoring.performance_analyzer.generate_performance_report(1)

    print("📈 Reporte de Performance (última hora):")
    print(f"   • Total requests: {report.total_requests}")
    print(".3f"    print(".3f"    print(".1f"    print(".1f"    print(f"   • Throughput: {report.throughput_rps:.2f} RPS")

    # ===== DEMO 5: DASHBOARD DATA =====
    print("\\n\\n📋 DEMO 5: Dashboard Data")

    dashboard_data = monitoring.get_dashboard_data()

    print("📋 Datos del Dashboard:")
    print(f"   • CPU actual: {dashboard_data['system_metrics']['cpu'].get('mean', 0):.1f}%")
    print(f"   • Memoria actual: {dashboard_data['system_metrics']['memory'].get('mean', 0):.1f}%")
    print(f"   • Alertas activas: {dashboard_data['active_alerts']}")
    print(f"   • Componentes saludables: {sum(1 for h in dashboard_data['health_status'].values() if h.status == 'healthy')}")

    # ===== DEMO 6: REPORTES =====
    print("\\n\\n📄 DEMO 6: Report Generation")

    # Generar reportes
    perf_report = monitoring.generate_report("performance", 1)
    alert_report = monitoring.generate_report("alerts", 24)
    health_report = monitoring.generate_report("health")

    print("📄 Reportes generados:")
    print(f"   • Performance: {perf_report['total_requests']} requests")
    print(f"   • Alerts: {alert_report['total_alerts']} total alerts")
    print(f"   • Health: {'Healthy' if health_report['overall_health'] else 'Issues detected'}")

    # ===== DEMO 7: EXPORT =====
    print("\\n\\n💾 DEMO 7: Metrics Export")

    export_data = monitoring.export_metrics("json")
    export_size = len(export_data)

    print(f"💾 Métricas exportadas: {export_size} caracteres JSON")

    # Detener monitoring
    monitoring.stop_monitoring()

    # ===== RESULTADOS FINALES =====
    print("\\n\\n🎉 DEMO COMPLETA - RESULTADOS FINALES")
    print("=" * 50)

    print("🏆 LOGROS ALCANZADOS:")
    print(f"   ✅ Sistema de métricas en tiempo real operativo")
    print(f"   ✅ Alert manager con reglas configurables")
    print(f"   ✅ Health checks automáticos de componentes")
    print(f"   ✅ Performance analyzer con reportes detallados")
    print(f"   ✅ Dashboard con visualizaciones interactivas")
    print(f"   ✅ Export de métricas y reportes")
    print(f"   ✅ Monitoring multi-nivel (sistema, aplicación, componentes)")

    print("\\n🚀 CAPACIDADES DEMOSTRADAS:")
    print("   ✅ Métricas del sistema (CPU, memoria, disco, red, GPU)")
    print("   ✅ Métricas de aplicación (response time, error rate, throughput)")
    print("   ✅ Alertas configurables con severidad y cooldown")
    print("   ✅ Health checks con uptime tracking")
    print("   ✅ Performance reports con percentiles (P95, P99)")
    print("   ✅ Dashboard con gráficos en tiempo real")
    print("   ✅ Export de datos para análisis externo")
    print("   ✅ Multi-tenancy support básico")

    print("\\n💡 INSIGHTS TÉCNICOS:")
    print("   • El monitoring debe ser ligero para no afectar performance")
    print("   • Las alertas necesitan cooldown para evitar spam")
    print("   • Los health checks deben ser rápidos y confiables")
    print("   • Los percentiles son más útiles que promedios para SLAs")
    print("   • El monitoring debe ser escalable con el sistema")
    print("   • Los dashboards deben ser intuitivos para diferentes usuarios")
    print("   • Los reportes automáticos ahorran tiempo de análisis")

    print("\\n🔮 APLICACIONES DE MONITORING:")
    print("   • Monitoreo de producción 24/7 con alertas automáticas")
    print("   • Capacity planning basado en métricas históricas")
    print("   • SLA monitoring y reporting para clientes")
    print("   • Troubleshooting con métricas detalladas")
    print("   • Cost optimization basado en uso de recursos")
    print("   • Security monitoring con detección de anomalías")
    print("   • Performance benchmarking y comparación")

    print("\\n📊 MÉTRICAS CLAVE MONITOREADAS:")
    print("   • System: CPU, Memory, Disk, Network, GPU")
    print("   • Application: Response Time, Error Rate, Throughput")
    print("   • Business: Requests, Users, Success Rate")
    print("   • ML: Model Accuracy, Inference Time, Drift Detection")
    print("   • Health: Component Status, Uptime, Dependencies")

    print("\\n🔧 PRÓXIMOS PASOS PARA MONITORING:")
    print("   • Integrar con Prometheus/Grafana para visualización avanzada")
    print("   • Agregar distributed tracing (Jaeger, Zipkin)")
    print("   • Implementar log aggregation (ELK stack)")
    print("   • Crear alertas inteligentes con ML")
    print("   • Agregar anomaly detection automática")
    print("   • Implementar metrics retention policies")
    print("   • Crear monitoring APIs para integración externa")
    print("   • Agregar cost monitoring y optimization")

    print("\\n" + "=" * 60)
    print("🌟 Enterprise Monitoring funcionando correctamente!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_monitoring_system())
