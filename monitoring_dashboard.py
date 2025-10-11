#!/usr/bin/env python3
"""
Dashboard de Monitoreo - AEGIS Framework
Sistema de monitoreo y visualización en tiempo real para la infraestructura
de IA distribuida y colaborativa.

Características principales:
- Monitoreo en tiempo real de nodos P2P
- Visualización de métricas de rendimiento
- Dashboard web interactivo con WebSockets
- Alertas automáticas y notificaciones
- Análisis de tendencias y predicciones
- Integración con sistemas de logging
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import sqlite3
import threading
from datetime import datetime, timedelta
import statistics
import psutil
import socket
import requests
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np
from werkzeug.serving import make_server
import queue

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Tipos de métricas del sistema"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_LATENCY = "network_latency"
    DISK_IO = "disk_io"
    NODE_COUNT = "node_count"
    TRANSACTION_RATE = "transaction_rate"
    CONSENSUS_TIME = "consensus_time"
    MODEL_ACCURACY = "model_accuracy"
    FAULT_TOLERANCE = "fault_tolerance"
    SECURITY_EVENTS = "security_events"

class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class NodeStatus(Enum):
    """Estados de nodos"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"

@dataclass
class Metric:
    """Métrica del sistema"""
    metric_id: str
    metric_type: MetricType
    node_id: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alerta del sistema"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    node_id: str
    metric_type: Optional[MetricType]
    threshold_value: Optional[float]
    current_value: Optional[float]
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class NodeInfo:
    """Información de nodo"""
    node_id: str
    node_type: str
    status: NodeStatus
    ip_address: str
    port: int
    last_seen: float
    uptime: float
    version: str
    capabilities: List[str]
    current_load: float
    health_score: float

@dataclass
class SystemHealth:
    """Salud general del sistema"""
    overall_score: float
    active_nodes: int
    total_nodes: int
    critical_alerts: int
    warning_alerts: int
    avg_response_time: float
    system_uptime: float
    last_updated: float

class MetricsCollector:
    """Recolector de métricas del sistema"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.metrics_queue = queue.Queue()
        self.collection_interval = 5  # segundos
        self.running = False
        self.collection_thread = None
    
    def start_collection(self):
        """Inicia recolección de métricas"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info(f"📊 Recolección de métricas iniciada para {self.node_id}")
    
    def stop_collection(self):
        """Detiene recolección de métricas"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info(f"⏹️ Recolección de métricas detenida para {self.node_id}")
    
    def _collect_metrics_loop(self):
        """Loop principal de recolección"""
        while self.running:
            try:
                # Recolectar métricas del sistema
                self._collect_system_metrics()
                
                # Recolectar métricas de red
                self._collect_network_metrics()
                
                # Recolectar métricas de aplicación
                self._collect_application_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error recolectando métricas: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Recolecta métricas del sistema operativo"""
        try:
            current_time = time.time()
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = Metric(
                metric_id=f"cpu_{current_time}",
                metric_type=MetricType.CPU_USAGE,
                node_id=self.node_id,
                value=cpu_percent,
                unit="percent",
                timestamp=current_time,
                metadata={"cores": psutil.cpu_count()}
            )
            self.metrics_queue.put(cpu_metric)
            
            # Memoria
            memory = psutil.virtual_memory()
            memory_metric = Metric(
                metric_id=f"memory_{current_time}",
                metric_type=MetricType.MEMORY_USAGE,
                node_id=self.node_id,
                value=memory.percent,
                unit="percent",
                timestamp=current_time,
                metadata={
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used
                }
            )
            self.metrics_queue.put(memory_metric)
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_metric = Metric(
                metric_id=f"disk_{current_time}",
                metric_type=MetricType.DISK_IO,
                node_id=self.node_id,
                value=(disk.used / disk.total) * 100,
                unit="percent",
                timestamp=current_time,
                metadata={
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free
                }
            )
            self.metrics_queue.put(disk_metric)
            
        except Exception as e:
            logger.error(f"❌ Error recolectando métricas de sistema: {e}")
    
    def _collect_network_metrics(self):
        """Recolecta métricas de red"""
        try:
            current_time = time.time()
            
            # Latencia de red (ping a localhost como ejemplo)
            start_time = time.time()
            try:
                socket.create_connection(("127.0.0.1", 80), timeout=1)
                latency = (time.time() - start_time) * 1000  # ms
            except:
                latency = 1000  # timeout
            
            latency_metric = Metric(
                metric_id=f"latency_{current_time}",
                metric_type=MetricType.NETWORK_LATENCY,
                node_id=self.node_id,
                value=latency,
                unit="ms",
                timestamp=current_time
            )
            self.metrics_queue.put(latency_metric)
            
        except Exception as e:
            logger.error(f"❌ Error recolectando métricas de red: {e}")
    
    def _collect_application_metrics(self):
        """Recolecta métricas específicas de la aplicación"""
        try:
            current_time = time.time()
            
            # Simular métricas de consenso
            consensus_time = np.random.normal(2.5, 0.5)  # Tiempo promedio de consenso
            consensus_metric = Metric(
                metric_id=f"consensus_{current_time}",
                metric_type=MetricType.CONSENSUS_TIME,
                node_id=self.node_id,
                value=max(0.1, consensus_time),
                unit="seconds",
                timestamp=current_time
            )
            self.metrics_queue.put(consensus_metric)
            
            # Simular precisión del modelo
            model_accuracy = np.random.normal(0.92, 0.02)
            accuracy_metric = Metric(
                metric_id=f"accuracy_{current_time}",
                metric_type=MetricType.MODEL_ACCURACY,
                node_id=self.node_id,
                value=max(0.0, min(1.0, model_accuracy)),
                unit="ratio",
                timestamp=current_time
            )
            self.metrics_queue.put(accuracy_metric)
            
        except Exception as e:
            logger.error(f"❌ Error recolectando métricas de aplicación: {e}")
    
    def get_metrics(self) -> List[Metric]:
        """Obtiene métricas recolectadas"""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return metrics

class AlertManager:
    """Gestor de alertas del sistema"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[MetricType, Dict[str, Any]] = {}
        self.notification_callbacks: List[Callable] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configura reglas de alerta por defecto"""
        self.alert_rules = {
            MetricType.CPU_USAGE: {
                "warning_threshold": 70.0,
                "critical_threshold": 90.0,
                "operator": "greater_than"
            },
            MetricType.MEMORY_USAGE: {
                "warning_threshold": 80.0,
                "critical_threshold": 95.0,
                "operator": "greater_than"
            },
            MetricType.NETWORK_LATENCY: {
                "warning_threshold": 100.0,
                "critical_threshold": 500.0,
                "operator": "greater_than"
            },
            MetricType.MODEL_ACCURACY: {
                "warning_threshold": 0.85,
                "critical_threshold": 0.75,
                "operator": "less_than"
            }
        }
    
    def add_notification_callback(self, callback: Callable):
        """Agrega callback para notificaciones"""
        self.notification_callbacks.append(callback)
    
    def evaluate_metric(self, metric: Metric):
        """Evalúa métrica contra reglas de alerta"""
        try:
            if metric.metric_type not in self.alert_rules:
                return
            
            rule = self.alert_rules[metric.metric_type]
            operator = rule["operator"]
            warning_threshold = rule["warning_threshold"]
            critical_threshold = rule["critical_threshold"]
            
            alert_level = None
            threshold_value = None
            
            if operator == "greater_than":
                if metric.value >= critical_threshold:
                    alert_level = AlertLevel.CRITICAL
                    threshold_value = critical_threshold
                elif metric.value >= warning_threshold:
                    alert_level = AlertLevel.WARNING
                    threshold_value = warning_threshold
            elif operator == "less_than":
                if metric.value <= critical_threshold:
                    alert_level = AlertLevel.CRITICAL
                    threshold_value = critical_threshold
                elif metric.value <= warning_threshold:
                    alert_level = AlertLevel.WARNING
                    threshold_value = warning_threshold
            
            if alert_level:
                self._create_alert(metric, alert_level, threshold_value)
                
        except Exception as e:
            logger.error(f"❌ Error evaluando métrica: {e}")
    
    def _create_alert(self, metric: Metric, level: AlertLevel, threshold_value: float):
        """Crea nueva alerta"""
        try:
            alert_id = f"{metric.node_id}_{metric.metric_type.value}_{int(metric.timestamp)}"
            
            # Evitar alertas duplicadas recientes
            if alert_id in self.alerts:
                return
            
            title = f"{metric.metric_type.value.replace('_', ' ').title()} {level.value.title()}"
            message = f"Node {metric.node_id}: {metric.metric_type.value} is {metric.value:.2f} {metric.unit} (threshold: {threshold_value:.2f})"
            
            alert = Alert(
                alert_id=alert_id,
                level=level,
                title=title,
                message=message,
                node_id=metric.node_id,
                metric_type=metric.metric_type,
                threshold_value=threshold_value,
                current_value=metric.value,
                timestamp=metric.timestamp
            )
            
            self.alerts[alert_id] = alert
            
            # Notificar callbacks
            for callback in self.notification_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"❌ Error en callback de notificación: {e}")
            
            logger.warning(f"🚨 {level.value.upper()}: {message}")
            
        except Exception as e:
            logger.error(f"❌ Error creando alerta: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtiene alertas activas"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconoce una alerta"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logger.info(f"✅ Alerta reconocida: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resuelve una alerta"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            logger.info(f"✅ Alerta resuelta: {alert_id}")
            return True
        return False

class NodeManager:
    """Gestor de nodos del sistema"""
    
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_timeout = 60  # segundos
    
    def register_node(self, node_info: NodeInfo):
        """Registra un nodo"""
        self.nodes[node_info.node_id] = node_info
        logger.info(f"🔗 Nodo registrado: {node_info.node_id}")
    
    def update_node_status(self, node_id: str, status: NodeStatus, health_score: float = None):
        """Actualiza estado de nodo"""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            self.nodes[node_id].last_seen = time.time()
            if health_score is not None:
                self.nodes[node_id].health_score = health_score
    
    def get_active_nodes(self) -> List[NodeInfo]:
        """Obtiene nodos activos"""
        current_time = time.time()
        active_nodes = []
        
        for node in self.nodes.values():
            if (current_time - node.last_seen) < self.node_timeout:
                if node.status == NodeStatus.OFFLINE:
                    node.status = NodeStatus.ONLINE
                active_nodes.append(node)
            else:
                node.status = NodeStatus.OFFLINE
        
        return active_nodes
    
    def get_system_health(self) -> SystemHealth:
        """Calcula salud general del sistema"""
        active_nodes = self.get_active_nodes()
        total_nodes = len(self.nodes)
        
        # Calcular puntuación promedio de salud
        if active_nodes:
            avg_health = statistics.mean([node.health_score for node in active_nodes])
        else:
            avg_health = 0.0
        
        return SystemHealth(
            overall_score=avg_health,
            active_nodes=len(active_nodes),
            total_nodes=total_nodes,
            critical_alerts=0,  # Se actualizará desde AlertManager
            warning_alerts=0,   # Se actualizará desde AlertManager
            avg_response_time=0.0,  # Se calculará desde métricas
            system_uptime=time.time(),  # Simplificado
            last_updated=time.time()
        )

class DashboardServer:
    """Servidor web del dashboard"""
    
    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'aegis_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Componentes del sistema
        self.metrics_collector = MetricsCollector("dashboard_node")
        self.alert_manager = AlertManager()
        self.node_manager = NodeManager()
        
        # Almacenamiento de datos
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.server_thread = None
        self.server = None
        
        self._setup_routes()
        self._setup_socketio_events()
        self._setup_alert_notifications()
    
    def _setup_routes(self):
        """Configura rutas HTTP"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            try:
                # Obtener métricas recientes
                metrics_data = {}
                for metric_type, history in self.metrics_history.items():
                    if history:
                        recent_metrics = list(history)[-50:]  # Últimas 50
                        metrics_data[metric_type] = [
                            {
                                'timestamp': m.timestamp,
                                'value': m.value,
                                'node_id': m.node_id,
                                'unit': m.unit
                            } for m in recent_metrics
                        ]
                
                return jsonify(metrics_data)
                
            except Exception as e:
                logger.error(f"❌ Error obteniendo métricas: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            try:
                alerts = self.alert_manager.get_active_alerts()
                return jsonify([asdict(alert) for alert in alerts])
            except Exception as e:
                logger.error(f"❌ Error obteniendo alertas: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/nodes')
        def get_nodes():
            try:
                nodes = self.node_manager.get_active_nodes()
                return jsonify([asdict(node) for node in nodes])
            except Exception as e:
                logger.error(f"❌ Error obteniendo nodos: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/health')
        def get_health():
            try:
                health = self.node_manager.get_system_health()
                
                # Actualizar con datos de alertas
                active_alerts = self.alert_manager.get_active_alerts()
                health.critical_alerts = len([a for a in active_alerts if a.level == AlertLevel.CRITICAL])
                health.warning_alerts = len([a for a in active_alerts if a.level == AlertLevel.WARNING])
                
                return jsonify(asdict(health))
            except Exception as e:
                logger.error(f"❌ Error obteniendo salud del sistema: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            try:
                success = self.alert_manager.acknowledge_alert(alert_id)
                return jsonify({"success": success})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _setup_socketio_events(self):
        """Configura eventos de WebSocket"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"🔌 Cliente conectado: {request.sid}")
            join_room('dashboard')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"🔌 Cliente desconectado: {request.sid}")
            leave_room('dashboard')
        
        @self.socketio.on('subscribe_metrics')
        def handle_subscribe_metrics(data):
            metric_types = data.get('metric_types', [])
            logger.info(f"📊 Cliente suscrito a métricas: {metric_types}")
    
    def _setup_alert_notifications(self):
        """Configura notificaciones de alertas"""
        def notify_alert(alert: Alert):
            self.socketio.emit('new_alert', asdict(alert), room='dashboard')
        
        self.alert_manager.add_notification_callback(notify_alert)
    
    def start_server(self):
        """Inicia servidor del dashboard"""
        try:
            # Iniciar recolección de métricas
            self.metrics_collector.start_collection()
            
            # Registrar nodo del dashboard
            dashboard_node = NodeInfo(
                node_id="dashboard_node",
                node_type="dashboard",
                status=NodeStatus.ONLINE,
                ip_address=self.host,
                port=self.port,
                last_seen=time.time(),
                uptime=0.0,
                version="1.0.0",
                capabilities=["monitoring", "alerting"],
                current_load=0.0,
                health_score=1.0
            )
            self.node_manager.register_node(dashboard_node)
            
            # Iniciar servidor web
            self.server = make_server(self.host, self.port, self.app, threaded=True)
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Iniciar loop de procesamiento
            self._start_processing_loop()
            
            logger.info(f"🌐 Dashboard iniciado en http://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ Error iniciando servidor: {e}")
    
    def _run_server(self):
        """Ejecuta servidor web"""
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error(f"❌ Error en servidor web: {e}")
    
    def _start_processing_loop(self):
        """Inicia loop de procesamiento de datos"""
        def processing_loop():
            while True:
                try:
                    # Procesar métricas nuevas
                    new_metrics = self.metrics_collector.get_metrics()
                    
                    for metric in new_metrics:
                        # Almacenar en historial
                        self.metrics_history[metric.metric_type.value].append(metric)
                        
                        # Evaluar alertas
                        self.alert_manager.evaluate_metric(metric)
                        
                        # Emitir a clientes WebSocket
                        self.socketio.emit('new_metric', asdict(metric), room='dashboard')
                    
                    # Actualizar estado de nodos
                    self.node_manager.update_node_status(
                        "dashboard_node", 
                        NodeStatus.ONLINE, 
                        1.0
                    )
                    
                    time.sleep(1)  # Procesar cada segundo
                    
                except Exception as e:
                    logger.error(f"❌ Error en loop de procesamiento: {e}")
                    time.sleep(5)
        
        processing_thread = threading.Thread(target=processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
    
    def stop_server(self):
        """Detiene servidor del dashboard"""
        try:
            self.metrics_collector.stop_collection()
            
            if self.server:
                self.server.shutdown()
            
            logger.info("⏹️ Dashboard detenido")
            
        except Exception as e:
            logger.error(f"❌ Error deteniendo servidor: {e}")
    
    def add_external_node(self, node_info: NodeInfo):
        """Agrega nodo externo al monitoreo"""
        self.node_manager.register_node(node_info)
    
    def simulate_distributed_system(self):
        """Simula sistema distribuido para demostración"""
        try:
            # Simular nodos adicionales
            for i in range(3):
                node_info = NodeInfo(
                    node_id=f"node_{i+1}",
                    node_type="worker",
                    status=NodeStatus.ONLINE,
                    ip_address=f"192.168.1.{100+i}",
                    port=8000 + i,
                    last_seen=time.time(),
                    uptime=time.time() - (i * 3600),  # Diferentes tiempos de inicio
                    version="1.0.0",
                    capabilities=["consensus", "learning", "storage"],
                    current_load=np.random.uniform(0.1, 0.8),
                    health_score=np.random.uniform(0.8, 1.0)
                )
                self.node_manager.register_node(node_info)
            
            logger.info("🎭 Sistema distribuido simulado")
            
        except Exception as e:
            logger.error(f"❌ Error simulando sistema: {e}")

# Template HTML para el dashboard
DASHBOARD_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS Framework - Dashboard de Monitoreo</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card { transition: all 0.3s ease; }
        .metric-card:hover { transform: translateY(-2px); }
        .alert-critical { border-left: 4px solid #ef4444; }
        .alert-warning { border-left: 4px solid #f59e0b; }
        .alert-info { border-left: 4px solid #3b82f6; }
        .node-online { color: #10b981; }
        .node-offline { color: #ef4444; }
        .node-degraded { color: #f59e0b; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🛡️ AEGIS Framework</h1>
            <p class="text-gray-600">Dashboard de Monitoreo en Tiempo Real</p>
        </div>

        <!-- System Health Overview -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6 metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Salud del Sistema</p>
                        <p class="text-2xl font-bold text-green-600" id="system-health">--</p>
                    </div>
                    <div class="text-green-500">
                        <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                        </svg>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6 metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Nodos Activos</p>
                        <p class="text-2xl font-bold text-blue-600" id="active-nodes">--</p>
                    </div>
                    <div class="text-blue-500">
                        <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6 metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Alertas Críticas</p>
                        <p class="text-2xl font-bold text-red-600" id="critical-alerts">--</p>
                    </div>
                    <div class="text-red-500">
                        <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                        </svg>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6 metric-card">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Tiempo de Respuesta</p>
                        <p class="text-2xl font-bold text-purple-600" id="response-time">--</p>
                    </div>
                    <div class="text-purple-500">
                        <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"></path>
                        </svg>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📊 Uso de CPU</h3>
                <div id="cpu-chart" style="height: 300px;"></div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">💾 Uso de Memoria</h3>
                <div id="memory-chart" style="height: 300px;"></div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">🌐 Latencia de Red</h3>
                <div id="latency-chart" style="height: 300px;"></div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">🤖 Precisión del Modelo</h3>
                <div id="accuracy-chart" style="height: 300px;"></div>
            </div>
        </div>

        <!-- Nodes and Alerts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">🔗 Estado de Nodos</h3>
                <div id="nodes-list" class="space-y-3">
                    <!-- Nodes will be populated here -->
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">🚨 Alertas Activas</h3>
                <div id="alerts-list" class="space-y-3">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Chart data storage
        const chartData = {
            cpu: { x: [], y: [] },
            memory: { x: [], y: [] },
            latency: { x: [], y: [] },
            accuracy: { x: [], y: [] }
        };

        // Initialize charts
        function initCharts() {
            const chartConfig = {
                displayModeBar: false,
                responsive: true
            };

            // CPU Chart
            Plotly.newPlot('cpu-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'CPU Usage',
                line: { color: '#3b82f6' }
            }], {
                title: '',
                xaxis: { title: 'Tiempo' },
                yaxis: { title: 'Porcentaje (%)' },
                margin: { t: 20 }
            }, chartConfig);

            // Memory Chart
            Plotly.newPlot('memory-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Memory Usage',
                line: { color: '#10b981' }
            }], {
                title: '',
                xaxis: { title: 'Tiempo' },
                yaxis: { title: 'Porcentaje (%)' },
                margin: { t: 20 }
            }, chartConfig);

            // Latency Chart
            Plotly.newPlot('latency-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Network Latency',
                line: { color: '#f59e0b' }
            }], {
                title: '',
                xaxis: { title: 'Tiempo' },
                yaxis: { title: 'Latencia (ms)' },
                margin: { t: 20 }
            }, chartConfig);

            // Accuracy Chart
            Plotly.newPlot('accuracy-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Model Accuracy',
                line: { color: '#8b5cf6' }
            }], {
                title: '',
                xaxis: { title: 'Tiempo' },
                yaxis: { title: 'Precisión', range: [0, 1] },
                margin: { t: 20 }
            }, chartConfig);
        }

        // Update chart with new data
        function updateChart(chartId, data) {
            const maxPoints = 50;
            
            if (data.x.length > maxPoints) {
                data.x = data.x.slice(-maxPoints);
                data.y = data.y.slice(-maxPoints);
            }

            Plotly.redraw(chartId, [{
                x: data.x,
                y: data.y,
                type: 'scatter',
                mode: 'lines+markers'
            }]);
        }

        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to dashboard server');
            loadInitialData();
        });

        socket.on('new_metric', function(metric) {
            const timestamp = new Date(metric.timestamp * 1000);
            
            switch(metric.metric_type) {
                case 'cpu_usage':
                    chartData.cpu.x.push(timestamp);
                    chartData.cpu.y.push(metric.value);
                    updateChart('cpu-chart', chartData.cpu);
                    break;
                case 'memory_usage':
                    chartData.memory.x.push(timestamp);
                    chartData.memory.y.push(metric.value);
                    updateChart('memory-chart', chartData.memory);
                    break;
                case 'network_latency':
                    chartData.latency.x.push(timestamp);
                    chartData.latency.y.push(metric.value);
                    updateChart('latency-chart', chartData.latency);
                    break;
                case 'model_accuracy':
                    chartData.accuracy.x.push(timestamp);
                    chartData.accuracy.y.push(metric.value);
                    updateChart('accuracy-chart', chartData.accuracy);
                    break;
            }
        });

        socket.on('new_alert', function(alert) {
            console.log('New alert:', alert);
            loadAlerts();
        });

        // Load initial data
        function loadInitialData() {
            loadHealth();
            loadNodes();
            loadAlerts();
            loadMetrics();
        }

        function loadHealth() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-health').textContent = (data.overall_score * 100).toFixed(1) + '%';
                    document.getElementById('active-nodes').textContent = data.active_nodes + '/' + data.total_nodes;
                    document.getElementById('critical-alerts').textContent = data.critical_alerts;
                    document.getElementById('response-time').textContent = data.avg_response_time.toFixed(1) + 'ms';
                });
        }

        function loadNodes() {
            fetch('/api/nodes')
                .then(response => response.json())
                .then(nodes => {
                    const nodesList = document.getElementById('nodes-list');
                    nodesList.innerHTML = '';
                    
                    nodes.forEach(node => {
                        const nodeElement = document.createElement('div');
                        nodeElement.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
                        
                        const statusClass = node.status === 'online' ? 'node-online' : 
                                          node.status === 'offline' ? 'node-offline' : 'node-degraded';
                        
                        nodeElement.innerHTML = `
                            <div>
                                <p class="font-medium">${node.node_id}</p>
                                <p class="text-sm text-gray-600">${node.ip_address}:${node.port}</p>
                            </div>
                            <div class="text-right">
                                <p class="font-medium ${statusClass}">${node.status.toUpperCase()}</p>
                                <p class="text-sm text-gray-600">Health: ${(node.health_score * 100).toFixed(0)}%</p>
                            </div>
                        `;
                        
                        nodesList.appendChild(nodeElement);
                    });
                });
        }

        function loadAlerts() {
            fetch('/api/alerts')
                .then(response => response.json())
                .then(alerts => {
                    const alertsList = document.getElementById('alerts-list');
                    alertsList.innerHTML = '';
                    
                    if (alerts.length === 0) {
                        alertsList.innerHTML = '<p class="text-gray-500 text-center py-4">No hay alertas activas</p>';
                        return;
                    }
                    
                    alerts.forEach(alert => {
                        const alertElement = document.createElement('div');
                        const alertClass = `alert-${alert.level}`;
                        
                        alertElement.className = `p-3 bg-gray-50 rounded-lg ${alertClass}`;
                        alertElement.innerHTML = `
                            <div class="flex items-start justify-between">
                                <div>
                                    <p class="font-medium">${alert.title}</p>
                                    <p class="text-sm text-gray-600">${alert.message}</p>
                                    <p class="text-xs text-gray-500 mt-1">${new Date(alert.timestamp * 1000).toLocaleString()}</p>
                                </div>
                                <button onclick="acknowledgeAlert('${alert.alert_id}')" 
                                        class="text-sm bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600">
                                    Reconocer
                                </button>
                            </div>
                        `;
                        
                        alertsList.appendChild(alertElement);
                    });
                });
        }

        function loadMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(metrics => {
                    // Process historical metrics for charts
                    Object.keys(metrics).forEach(metricType => {
                        const metricData = metrics[metricType];
                        
                        switch(metricType) {
                            case 'cpu_usage':
                                chartData.cpu.x = metricData.map(m => new Date(m.timestamp * 1000));
                                chartData.cpu.y = metricData.map(m => m.value);
                                updateChart('cpu-chart', chartData.cpu);
                                break;
                            case 'memory_usage':
                                chartData.memory.x = metricData.map(m => new Date(m.timestamp * 1000));
                                chartData.memory.y = metricData.map(m => m.value);
                                updateChart('memory-chart', chartData.memory);
                                break;
                            case 'network_latency':
                                chartData.latency.x = metricData.map(m => new Date(m.timestamp * 1000));
                                chartData.latency.y = metricData.map(m => m.value);
                                updateChart('latency-chart', chartData.latency);
                                break;
                            case 'model_accuracy':
                                chartData.accuracy.x = metricData.map(m => new Date(m.timestamp * 1000));
                                chartData.accuracy.y = metricData.map(m => m.value);
                                updateChart('accuracy-chart', chartData.accuracy);
                                break;
                        }
                    });
                });
        }

        function acknowledgeAlert(alertId) {
            fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadAlerts();
                }
            });
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            
            // Refresh data periodically
            setInterval(loadHealth, 5000);
            setInterval(loadNodes, 10000);
            setInterval(loadAlerts, 15000);
        });
    </script>
</body>
</html>
'''

# Función principal para testing
async def main():
    """Función principal para pruebas"""
    try:
        # Crear directorio de templates si no existe
        import os
        os.makedirs('templates', exist_ok=True)
        
        # Escribir template HTML
        with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
            f.write(DASHBOARD_HTML_TEMPLATE)
        
        # Crear servidor del dashboard
        dashboard = DashboardServer(host="localhost", port=5000)
        
        # Simular sistema distribuido
        dashboard.simulate_distributed_system()
        
        # Iniciar servidor
        dashboard.start_server()
        
        print("🌐 Dashboard iniciado en http://localhost:5000")
        print("📊 Presiona Ctrl+C para detener")
        
        # Mantener servidor corriendo
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Deteniendo dashboard...")
            dashboard.stop_server()
            
    except Exception as e:
        logger.error(f"❌ Error en main: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    
def start_dashboard(config: dict):
    """Adapter a nivel de módulo para iniciar el dashboard.
    Inicia DashboardServer y devuelve la instancia.
    """
    try:
        host = config.get("host", "127.0.0.1")
        port = int(config.get("dashboard_port", 8080))
        server = DashboardServer(host=host, port=port)
        server.start_server()
        logger.info(f"🌐 Dashboard disponible en http://{host}:{port}")
        return server
    except Exception as e:
        logger.error(f"❌ No se pudo iniciar el dashboard: {e}")
        return None