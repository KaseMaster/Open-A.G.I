#!/usr/bin/env python3
"""
AEGIS - Dashboard Web Interactivo
Sistema de monitoreo y control web en tiempo real para AEGIS
"""

import asyncio
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psutil
import logging

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    from loguru import logger
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Dependencias faltantes para dashboard web: {e}")
    Flask = None
    SocketIO = None

class SystemMonitor:
    """Monitor de sistema en tiempo real"""
    
    def __init__(self):
        self.running = False
        self.data_history = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network': [],
            'timestamps': []
        }
        self.max_history = 100
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            stats = {
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Actualizar historial
            self._update_history(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del sistema: {e}")
            return {}
    
    def _update_history(self, stats: Dict[str, Any]):
        """Actualiza el historial de datos"""
        timestamp = datetime.now()
        
        self.data_history['timestamps'].append(timestamp.isoformat())
        self.data_history['cpu'].append(stats['cpu']['percent'])
        self.data_history['memory'].append(stats['memory']['percent'])
        self.data_history['disk'].append(stats['disk']['percent'])
        
        # Mantener solo los últimos N registros
        for key in self.data_history:
            if len(self.data_history[key]) > self.max_history:
                self.data_history[key] = self.data_history[key][-self.max_history:]
    
    def get_history(self) -> Dict[str, List]:
        """Obtiene el historial de datos"""
        return self.data_history.copy()

class AEGISServicesMonitor:
    """Monitor de servicios AEGIS"""
    
    def __init__(self):
        self.services_status = {}
        
    def get_services_status(self) -> Dict[str, Any]:
        """Obtiene el estado de los servicios AEGIS"""
        services = {
            'tor': self._check_service_status('tor'),
            'p2p': self._check_service_status('p2p'),
            'crypto': self._check_service_status('crypto'),
            'consensus': self._check_service_status('consensus'),
            'monitoring': self._check_service_status('monitoring'),
            'api_server': self._check_service_status('api_server'),
            'metrics': self._check_service_status('metrics'),
            'alerts': self._check_service_status('alerts')
        }
        
        return {
            'services': services,
            'total_services': len(services),
            'active_services': sum(1 for s in services.values() if s['status'] == 'active'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_service_status(self, service_name: str) -> Dict[str, Any]:
        """Verifica el estado de un servicio específico"""
        # Simulación del estado del servicio
        # En implementación real, verificaría el estado real del servicio
        return {
            'status': 'active',  # active, inactive, error
            'uptime': '2h 15m',
            'memory_usage': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB",
            'last_check': datetime.now().isoformat()
        }

class MetricsDatabase:
    """Base de datos para métricas del dashboard"""
    
    def __init__(self, db_path: str = "aegis_dashboard.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_percent REAL,
                        network_bytes_sent INTEGER,
                        network_bytes_recv INTEGER
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS service_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        service_name TEXT,
                        event_type TEXT,
                        message TEXT,
                        severity TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
    
    def store_metrics(self, metrics: Dict[str, Any]):
        """Almacena métricas en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics 
                    (cpu_percent, memory_percent, disk_percent, network_bytes_sent, network_bytes_recv)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metrics['cpu']['percent'],
                    metrics['memory']['percent'],
                    metrics['disk']['percent'],
                    metrics['network']['bytes_sent'],
                    metrics['network']['bytes_recv']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error almacenando métricas: {e}")
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Obtiene historial de métricas"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM system_metrics 
                    WHERE timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                """.format(hours))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error obteniendo historial de métricas: {e}")
            return []

class AEGISWebDashboard:
    """Dashboard web principal de AEGIS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = None
        self.socketio = None
        self.system_monitor = SystemMonitor()
        self.services_monitor = AEGISServicesMonitor()
        self.metrics_db = MetricsDatabase()
        self.running = False
        
        if Flask and SocketIO:
            self._init_flask_app()
        else:
            logger.error("Flask o SocketIO no disponibles")
    
    def _init_flask_app(self):
        """Inicializa la aplicación Flask"""
        if Flask is None or SocketIO is None:
            logger.error("Flask o SocketIO no disponibles")
            return
            
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'aegis-dashboard-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """Configura las rutas de la aplicación"""
        if self.app is None:
            return
            
        # Verificar que las funciones necesarias estén disponibles
        if jsonify is None or request is None or send_from_directory is None:
            logger.error("Funciones Flask no disponibles")
            return
            
        @self.app.route('/')
        def index():
            return self._render_dashboard()
        
        @self.app.route('/api/system/stats')
        def system_stats():
            return jsonify(self.system_monitor.get_system_stats())
        
        @self.app.route('/api/services/status')
        def services_status():
            return jsonify(self.services_monitor.get_services_status())
        
        @self.app.route('/api/metrics/history')
        def metrics_history():
            hours = request.args.get('hours', 24, type=int)
            return jsonify(self.metrics_db.get_metrics_history(hours))
        
        @self.app.route('/api/charts/system')
        def system_charts():
            return jsonify(self._generate_system_charts())
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            return send_from_directory('static', filename)
    
    def _setup_socketio_events(self):
        """Configura eventos de SocketIO"""
        if self.socketio is None:
            return
            
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Cliente conectado al dashboard")
            emit('status', {'msg': 'Conectado al dashboard AEGIS'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Cliente desconectado del dashboard")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            stats = self.system_monitor.get_system_stats()
            services = self.services_monitor.get_services_status()
            emit('system_update', {
                'system': stats,
                'services': services,
                'timestamp': datetime.now().isoformat()
            })
    
    def _render_dashboard(self) -> str:
        """Renderiza el dashboard principal"""
        html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 10px;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #34495e;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        .metric:last-child { border-bottom: none; }
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-active { background: #27ae60; }
        .status-inactive { background: #e74c3c; }
        .status-warning { background: #f39c12; }
        .chart-container {
            height: 400px;
            margin-top: 20px;
        }
        .services-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .service-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #27ae60;
        }
        .service-card.inactive { border-left-color: #e74c3c; }
        .service-card.warning { border-left-color: #f39c12; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .updating { animation: pulse 1s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ AEGIS Dashboard</h1>
            <div class="status-bar">
                <div>Estado: <span id="connection-status">Conectando...</span></div>
                <div>Última actualización: <span id="last-update">--</span></div>
                <div>Servicios activos: <span id="active-services">--</span></div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>📊 Sistema</h3>
                <div class="metric">
                    <span>CPU</span>
                    <span class="metric-value" id="cpu-usage">--</span>
                </div>
                <div class="metric">
                    <span>Memoria</span>
                    <span class="metric-value" id="memory-usage">--</span>
                </div>
                <div class="metric">
                    <span>Disco</span>
                    <span class="metric-value" id="disk-usage">--</span>
                </div>
                <div class="metric">
                    <span>Red (Enviado)</span>
                    <span class="metric-value" id="network-sent">--</span>
                </div>
                <div class="metric">
                    <span>Red (Recibido)</span>
                    <span class="metric-value" id="network-recv">--</span>
                </div>
            </div>

            <div class="card">
                <h3>🔧 Servicios AEGIS</h3>
                <div class="services-grid" id="services-container">
                    <!-- Servicios se cargan dinámicamente -->
                </div>
            </div>
        </div>

        <div class="card">
            <h3>📈 Gráficos en Tiempo Real</h3>
            <div class="chart-container" id="system-chart"></div>
        </div>
    </div>

    <script>
        const socket = io();
        let systemChart = null;

        // Conexión SocketIO
        socket.on('connect', function() {
            document.getElementById('connection-status').textContent = 'Conectado';
            document.getElementById('connection-status').style.color = '#27ae60';
            requestUpdate();
        });

        socket.on('disconnect', function() {
            document.getElementById('connection-status').textContent = 'Desconectado';
            document.getElementById('connection-status').style.color = '#e74c3c';
        });

        socket.on('system_update', function(data) {
            updateSystemMetrics(data.system);
            updateServices(data.services);
            updateChart(data.system);
            document.getElementById('last-update').textContent = 
                new Date(data.timestamp).toLocaleTimeString();
        });

        function requestUpdate() {
            socket.emit('request_update');
        }

        function updateSystemMetrics(system) {
            document.getElementById('cpu-usage').textContent = system.cpu.percent + '%';
            document.getElementById('memory-usage').textContent = system.memory.percent + '%';
            document.getElementById('disk-usage').textContent = system.disk.percent.toFixed(1) + '%';
            document.getElementById('network-sent').textContent = formatBytes(system.network.bytes_sent);
            document.getElementById('network-recv').textContent = formatBytes(system.network.bytes_recv);
        }

        function updateServices(services) {
            const container = document.getElementById('services-container');
            const activeCount = document.getElementById('active-services');
            
            activeCount.textContent = services.active_services + '/' + services.total_services;
            
            container.innerHTML = '';
            Object.entries(services.services).forEach(([name, service]) => {
                const serviceCard = document.createElement('div');
                serviceCard.className = `service-card ${service.status}`;
                serviceCard.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 5px;">
                        <span class="status-indicator status-${service.status}"></span>
                        ${name.toUpperCase()}
                    </div>
                    <div style="font-size: 0.9em; color: #666;">
                        Uptime: ${service.uptime}<br>
                        Memoria: ${service.memory_usage}
                    </div>
                `;
                container.appendChild(serviceCard);
            });
        }

        function updateChart(system) {
            if (!systemChart) {
                initChart();
            }
            
            const now = new Date();
            Plotly.extendTraces('system-chart', {
                x: [[now], [now], [now]],
                y: [[system.cpu.percent], [system.memory.percent], [system.disk.percent]]
            }, [0, 1, 2]);
        }

        function initChart() {
            const trace1 = {
                x: [],
                y: [],
                name: 'CPU %',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#e74c3c' }
            };
            
            const trace2 = {
                x: [],
                y: [],
                name: 'Memoria %',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#3498db' }
            };
            
            const trace3 = {
                x: [],
                y: [],
                name: 'Disco %',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#f39c12' }
            };

            const layout = {
                title: 'Uso de Recursos del Sistema',
                xaxis: { title: 'Tiempo' },
                yaxis: { title: 'Porcentaje (%)', range: [0, 100] },
                showlegend: true
            };

            Plotly.newPlot('system-chart', [trace1, trace2, trace3], layout);
            systemChart = true;
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // Actualizar cada 5 segundos
        setInterval(requestUpdate, 5000);
        
        // Inicializar gráfico al cargar
        setTimeout(initChart, 1000);
    </script>
</body>
</html>
        """
        return html_template
    
    def _generate_system_charts(self) -> Dict[str, Any]:
        """Genera datos para gráficos del sistema"""
        history = self.system_monitor.get_history()
        
        if not history['timestamps']:
            return {'error': 'No hay datos disponibles'}
        
        # Gráfico de CPU
        cpu_chart = {
            'data': [{
                'x': history['timestamps'],
                'y': history['cpu'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'CPU %',
                'line': {'color': '#e74c3c'}
            }],
            'layout': {
                'title': 'Uso de CPU',
                'xaxis': {'title': 'Tiempo'},
                'yaxis': {'title': 'Porcentaje (%)'}
            }
        }
        
        # Gráfico de Memoria
        memory_chart = {
            'data': [{
                'x': history['timestamps'],
                'y': history['memory'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Memoria %',
                'line': {'color': '#3498db'}
            }],
            'layout': {
                'title': 'Uso de Memoria',
                'xaxis': {'title': 'Tiempo'},
                'yaxis': {'title': 'Porcentaje (%)'}
            }
        }
        
        return {
            'cpu_chart': cpu_chart,
            'memory_chart': memory_chart,
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_background_monitoring(self):
        """Inicia el monitoreo en segundo plano"""
        self.running = True
        
        def monitoring_loop():
            while self.running:
                try:
                    # Obtener métricas del sistema
                    stats = self.system_monitor.get_system_stats()
                    
                    # Almacenar en base de datos
                    self.metrics_db.store_metrics(stats)
                    
                    # Emitir actualización via WebSocket
                    if self.socketio:
                        services = self.services_monitor.get_services_status()
                        self.socketio.emit('system_update', {
                            'system': stats,
                            'services': services,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    time.sleep(5)  # Actualizar cada 5 segundos
                    
                except Exception as e:
                    logger.error(f"Error en monitoreo: {e}")
                    time.sleep(10)
        
        # Ejecutar en hilo separado
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Monitoreo en segundo plano iniciado")
    
    async def start_dashboard(self):
        """Inicia el dashboard web"""
        if not self.app or not self.socketio:
            logger.error("Dashboard no puede iniciarse: Flask/SocketIO no disponibles")
            return
        
        try:
            # Iniciar monitoreo en segundo plano
            await self.start_background_monitoring()
            
            # Configuración del servidor
            host = self.config.get('host', '0.0.0.0')
            port = self.config.get('port', 8080)
            debug = self.config.get('debug', False)
            
            logger.info(f"Iniciando dashboard web en http://{host}:{port}")
            
            # Ejecutar servidor en hilo separado para no bloquear
            def run_server():
                self.socketio.run(
                    self.app,
                    host=host,
                    port=port,
                    debug=debug,
                    allow_unsafe_werkzeug=True
                )
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            logger.info("✅ Dashboard web iniciado correctamente")
            
        except Exception as e:
            logger.error(f"Error iniciando dashboard web: {e}")
            raise
    
    def stop_dashboard(self):
        """Detiene el dashboard web"""
        self.running = False
        logger.info("Dashboard web detenido")

# Variables globales
_dashboard_instance: Optional[AEGISWebDashboard] = None

async def start_web_dashboard(config: Dict[str, Any]):
    """Inicia el dashboard web de AEGIS"""
    global _dashboard_instance
    
    try:
        logger.info("Iniciando dashboard web interactivo de AEGIS...")
        
        _dashboard_instance = AEGISWebDashboard(config)
        await _dashboard_instance.start_dashboard()
        
        logger.info("Dashboard web iniciado correctamente")
        
    except Exception as e:
        logger.error(f"Error iniciando dashboard web: {e}")
        raise

async def stop_web_dashboard():
    """Detiene el dashboard web"""
    global _dashboard_instance
    
    if _dashboard_instance:
        _dashboard_instance.stop_dashboard()
        _dashboard_instance = None
        logger.info("Dashboard web detenido")

def get_dashboard_instance() -> Optional[AEGISWebDashboard]:
    """Obtiene la instancia del dashboard"""
    return _dashboard_instance

if __name__ == "__main__":
    # Demostración del dashboard
    
    config = {
        'host': '0.0.0.0',
        'port': 8080,
        'debug': True
    }
    
    async def demo():
        await start_web_dashboard(config)
        
        # Mantener ejecutándose
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await stop_web_dashboard()
            logger.info("Dashboard detenido por usuario")
    
    asyncio.run(demo())