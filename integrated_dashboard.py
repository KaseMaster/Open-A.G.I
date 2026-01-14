#!/usr/bin/env python3
"""
Dashboard de Monitoreo Integrado - AEGIS Framework
Dashboard actualizado para mostrar métricas reales de P2P, Knowledge Base y Heartbeat.

Integración con componentes reales:
- P2P Network Manager: estado de red, peers conectados, latencia
- Knowledge Base: entradas, ramas, sincronización
- Heartbeat System: estado de nodos, métricas de salud
- Crypto Framework: métricas de seguridad
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Importar componentes reales del framework
try:
    from p2p_network import start_network
    from distributed_knowledge_base import initialize_knowledge_base
    from distributed_heartbeat import initialize_heartbeat_system
    from crypto_framework import initialize_crypto
    P2P_AVAILABLE = True
except Exception as e:
    logging.warning(f"Componentes del framework no disponibles: {e}")
    P2P_AVAILABLE = False

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedDashboardServer:
    """Dashboard integrado con componentes reales del framework"""

    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'aegis_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Componentes del framework real
        self.p2p_manager = None
        self.knowledge_base = None
        self.heartbeat_manager = None
        self.crypto_engine = None

        # Estado del sistema
        self.system_metrics = {
            "p2p_active": False,
            "kb_active": False,
            "heartbeat_active": False,
            "crypto_active": False
        }

        self._setup_routes()
        self._setup_socketio_events()

        logger.info(f"🔧 Dashboard integrado inicializado en {host}:{port}")

    def _setup_routes(self):
        """Configura rutas HTTP con integración real"""

        @self.app.route('/')
        def dashboard():
            return render_template('integrated_dashboard.html')

        @self.app.route('/api/health')
        def get_integrated_health():
            """Obtiene salud del sistema integrada"""
            try:
                health_data = {
                    "overall_score": 1.0,
                    "active_nodes": 0,
                    "total_nodes": 0,
                    "critical_alerts": 0,
                    "avg_response_time": 0.0,
                    "components": {}
                }

                # Salud del sistema P2P
                if self.p2p_manager:
                    try:
                        p2p_status = asyncio.run(self.p2p_manager.get_network_status())
                        health_data["components"]["p2p"] = {
                            "active": True,
                            "connected_peers": p2p_status.get("connected_peers", 0),
                            "discovered_peers": p2p_status.get("discovered_peers", 0)
                        }
                        health_data["active_nodes"] += p2p_status.get("connected_peers", 0)
                        self.system_metrics["p2p_active"] = True
                    except Exception as e:
                        logger.error(f"Error obteniendo estado P2P: {e}")
                        health_data["components"]["p2p"] = {"active": False, "error": str(e)}

                # Salud de la Knowledge Base
                if self.knowledge_base:
                    try:
                        kb_status = self.knowledge_base.get_api_status()
                        health_data["components"]["knowledge_base"] = {
                            "active": True,
                            "total_entries": kb_status["total_entries"],
                            "total_branches": kb_status["total_branches"],
                            "sync_peers": kb_status["sync_peers"]
                        }
                        self.system_metrics["kb_active"] = True
                    except Exception as e:
                        logger.error(f"Error obteniendo estado KB: {e}")
                        health_data["components"]["knowledge_base"] = {"active": False, "error": str(e)}

                # Salud del Heartbeat System
                if self.heartbeat_manager:
                    try:
                        heartbeat_status = self.heartbeat_manager.get_heartbeat_status()
                        health_data["components"]["heartbeat"] = {
                            "active": True,
                            "healthy_nodes": heartbeat_status["healthy_nodes"],
                            "total_nodes": heartbeat_status["total_nodes"],
                            "overall_health": heartbeat_status["overall_health"]
                        }
                        health_data["active_nodes"] += heartbeat_status["healthy_nodes"]
                        health_data["total_nodes"] += heartbeat_status["total_nodes"]
                        self.system_metrics["heartbeat_active"] = True
                    except Exception as e:
                        logger.error(f"Error obteniendo estado heartbeat: {e}")
                        health_data["components"]["heartbeat"] = {"active": False, "error": str(e)}

                # Salud del Crypto Framework
                if self.crypto_engine:
                    try:
                        crypto_metrics = self.crypto_engine.get_security_metrics()
                        health_data["components"]["crypto"] = {
                            "active": True,
                            "security_level": crypto_metrics["security_level"],
                            "active_channels": crypto_metrics["active_channels"],
                            "known_peers": crypto_metrics["known_peers"]
                        }
                        self.system_metrics["crypto_active"] = True
                    except Exception as e:
                        logger.error(f"Error obteniendo métricas crypto: {e}")
                        health_data["components"]["crypto"] = {"active": False, "error": str(e)}

                # Calcular puntuación general
                active_components = sum(1 for comp in health_data["components"].values() if comp.get("active", False))
                total_components = len(health_data["components"])

                if total_components > 0:
                    health_data["overall_score"] = active_components / total_components

                return jsonify(health_data)

            except Exception as e:
                logger.error(f"Error obteniendo salud integrada: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/p2p')
        def get_p2p_status():
            """Obtiene estado detallado de P2P"""
            if not self.p2p_manager:
                return jsonify({"error": "P2P no inicializado"}), 503

            try:
                status = asyncio.run(self.p2p_manager.get_network_status())
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado P2P: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/knowledge-base')
        def get_knowledge_base_status():
            """Obtiene estado de Knowledge Base"""
            if not self.knowledge_base:
                return jsonify({"error": "Knowledge Base no inicializada"}), 503

            try:
                status = self.knowledge_base.get_api_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado KB: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/heartbeat')
        def get_heartbeat_status():
            """Obtiene estado de Heartbeat System"""
            if not self.heartbeat_manager:
                return jsonify({"error": "Heartbeat System no inicializado"}), 503

            try:
                status = self.heartbeat_manager.get_heartbeat_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado heartbeat: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/crypto')
        def get_crypto_metrics():
            """Obtiene métricas criptográficas"""
            if not self.crypto_engine:
                return jsonify({"error": "Crypto Framework no inicializado"}), 503

            try:
                metrics = self.crypto_engine.get_security_metrics()
                return jsonify(metrics)
            except Exception as e:
                logger.error(f"Error obteniendo métricas crypto: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/components')
        def get_components_status():
            """Obtiene estado de todos los componentes"""
            return jsonify(self.system_metrics)

    def _setup_socketio_events(self):
        """Configura eventos de WebSocket"""

        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"🔌 Cliente conectado: {request.sid}")
            self.socketio.emit('components_status', self.system_metrics, room=request.sid)

        @self.socketio.on('get_real_time_metrics')
        def handle_get_real_time_metrics():
            """Envía métricas en tiempo real"""
            try:
                metrics_data = {}

                # Métricas de P2P
                if self.p2p_manager:
                    p2p_status = asyncio.run(self.p2p_manager.get_network_status())
                    metrics_data["p2p"] = {
                        "connected_peers": p2p_status.get("connected_peers", 0),
                        "network_active": p2p_status.get("network_active", False),
                        "topology": p2p_status.get("topology", {})
                    }

                # Métricas de Knowledge Base
                if self.knowledge_base:
                    kb_status = self.knowledge_base.get_api_status()
                    metrics_data["knowledge_base"] = {
                        "total_entries": kb_status["total_entries"],
                        "total_branches": kb_status["total_branches"],
                        "formats_used": kb_status["formats_used"],
                        "sync_peers": kb_status["sync_peers"]
                    }

                # Métricas de Heartbeat
                if self.heartbeat_manager:
                    heartbeat_status = self.heartbeat_manager.get_heartbeat_status()
                    metrics_data["heartbeat"] = {
                        "healthy_nodes": heartbeat_status["healthy_nodes"],
                        "total_nodes": heartbeat_status["total_nodes"],
                        "overall_health": heartbeat_status["overall_health"],
                        "recovery_in_progress": heartbeat_status["recovery_in_progress"]
                    }

                # Métricas criptográficas
                if self.crypto_engine:
                    crypto_metrics = self.crypto_engine.get_security_metrics()
                    metrics_data["crypto"] = {
                        "security_level": crypto_metrics["security_level"],
                        "active_channels": crypto_metrics["active_channels"],
                        "known_peers": crypto_metrics["known_peers"]
                    }

                self.socketio.emit('real_time_metrics', metrics_data, room=request.sid)

            except Exception as e:
                logger.error(f"Error enviando métricas en tiempo real: {e}")

    async def initialize_components(self, config: Dict[str, Any]):
        """Inicializa componentes del framework real"""
        logger.info("🚀 Inicializando componentes del framework...")

        try:
            # Inicializar P2P Network
            if config.get("enable_p2p", True):
                logger.info("🔗 Inicializando P2P Network...")
                p2p_config = config.get("p2p", {})
                self.p2p_manager = start_network(p2p_config)
                self.system_metrics["p2p_active"] = True
                logger.info("✅ P2P Network inicializado")

            # Inicializar Knowledge Base
            if config.get("enable_knowledge_base", True):
                logger.info("📚 Inicializando Knowledge Base...")
                kb_config = config.get("knowledge_base", {})
                self.knowledge_base = initialize_knowledge_base(kb_config)
                self.system_metrics["kb_active"] = True
                logger.info("✅ Knowledge Base inicializado")

            # Inicializar Heartbeat System
            if config.get("enable_heartbeat", True):
                logger.info("💓 Inicializando Heartbeat System...")
                heartbeat_config = config.get("heartbeat", {})
                self.heartbeat_manager = initialize_heartbeat_system(heartbeat_config)
                await self.heartbeat_manager.start_heartbeat_system()
                self.system_metrics["heartbeat_active"] = True
                logger.info("✅ Heartbeat System inicializado")

            # Inicializar Crypto Framework
            if config.get("enable_crypto", True):
                logger.info("🔐 Inicializando Crypto Framework...")
                crypto_config = config.get("crypto", {})
                self.crypto_engine = initialize_crypto(crypto_config)
                self.system_metrics["crypto_active"] = True
                logger.info("✅ Crypto Framework inicializado")

            logger.info("🎉 Todos los componentes inicializados exitosamente")

        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            raise

    def start_server(self):
        """Inicia servidor del dashboard"""
        try:
            logger.info(f"🌐 Iniciando dashboard en http://{self.host}:{self.port}")

            # Iniciar servidor con SocketIO
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                allow_unsafe_werkzeug=True
            )

        except Exception as e:
            logger.error(f"❌ Error iniciando servidor: {e}")
            raise

    def stop_server(self):
        """Detiene servidor y componentes"""
        logger.info("🛑 Deteniendo dashboard y componentes...")

        # Detener componentes del framework
        if self.heartbeat_manager:
            asyncio.run(self.heartbeat_manager.stop_heartbeat_system())

        if self.crypto_engine:
            asyncio.run(self.crypto_engine.shutdown())

        logger.info("✅ Dashboard detenido")


# Template HTML actualizado para dashboard integrado
INTEGRATED_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS Framework - Dashboard Integrado</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card { transition: all 0.3s ease; }
        .metric-card:hover { transform: translateY(-2px); }
        .component-active { color: #10b981; }
        .component-inactive { color: #ef4444; }
        .status-healthy { background-color: #d1fae5; color: #065f46; }
        .status-degraded { background-color: #fef3c7; color: #92400e; }
        .status-failed { background-color: #fee2e2; color: #991b1b; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🛡️ AEGIS Framework</h1>
            <p class="text-gray-600">Dashboard Integrado - Métricas en Tiempo Real</p>
        </div>

        <!-- Component Status -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-white rounded-lg shadow-md p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">P2P Network</p>
                        <p class="text-lg font-bold component-active" id="p2p-status">🔗 Inactivo</p>
                    </div>
                    <div id="p2p-peers" class="text-sm text-gray-500">0 peers</div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Knowledge Base</p>
                        <p class="text-lg font-bold component-active" id="kb-status">📚 Inactivo</p>
                    </div>
                    <div id="kb-entries" class="text-sm text-gray-500">0 entradas</div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Heartbeat System</p>
                        <p class="text-lg font-bold component-active" id="heartbeat-status">💓 Inactivo</p>
                    </div>
                    <div id="heartbeat-health" class="text-sm text-gray-500">0% salud</div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm font-medium text-gray-600">Crypto Framework</p>
                        <p class="text-lg font-bold component-active" id="crypto-status">🔐 Inactivo</p>
                    </div>
                    <div id="crypto-channels" class="text-sm text-gray-500">0 canales</div>
                </div>
            </div>
        </div>

        <!-- Real-time Metrics -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📊 Estado de Red P2P</h3>
                <div id="p2p-metrics" class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">Peers Conectados:</span>
                        <span id="p2p-connected-peers" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Peers Descubiertos:</span>
                        <span id="p2p-discovered-peers" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Estado de Red:</span>
                        <span id="p2p-network-active" class="font-medium">--</span>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📚 Knowledge Base</h3>
                <div id="kb-metrics" class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">Entradas Totales:</span>
                        <span id="kb-total-entries" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Ramas Activas:</span>
                        <span id="kb-total-branches" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Peers de Sync:</span>
                        <span id="kb-sync-peers" class="font-medium">--</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">💓 Heartbeat System</h3>
                <div id="heartbeat-metrics" class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">Nodos Saludables:</span>
                        <span id="heartbeat-healthy-nodes" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Nodos Totales:</span>
                        <span id="heartbeat-total-nodes" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Salud General:</span>
                        <span id="heartbeat-overall-health" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Recuperaciones:</span>
                        <span id="heartbeat-recovery-progress" class="font-medium">--</span>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">🔐 Crypto Framework</h3>
                <div id="crypto-metrics" class="space-y-3">
                    <div class="flex justify-between">
                        <span class="text-gray-600">Nivel de Seguridad:</span>
                        <span id="crypto-security-level" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Canales Activos:</span>
                        <span id="crypto-active-channels" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Peers Conocidos:</span>
                        <span id="crypto-known-peers" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-600">Estado:</span>
                        <span id="crypto-status-indicator" class="font-medium">--</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Logs -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">📋 Logs del Sistema</h3>
            <div id="system-logs" class="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-64 overflow-y-auto">
                <!-- Logs will be populated here -->
            </div>
        </div>
    </div>

    <script>
        const socket = io();

        socket.on('connect', function() {
            console.log('Connected to integrated dashboard');
            updateRealTimeMetrics();
        });

        socket.on('components_status', function(status) {
            updateComponentStatus(status);
        });

        socket.on('real_time_metrics', function(metrics) {
            updateMetrics(metrics);
        });

        function updateComponentStatus(status) {
            // P2P Status
            const p2pStatus = document.getElementById('p2p-status');
            const p2pPeers = document.getElementById('p2p-peers');

            if (status.p2p_active) {
                p2pStatus.textContent = '🔗 Activo';
                p2pStatus.className = 'text-lg font-bold component-active';
                p2pPeers.textContent = 'Cargando...';
            } else {
                p2pStatus.textContent = '🔗 Inactivo';
                p2pStatus.className = 'text-lg font-bold component-inactive';
                p2pPeers.textContent = '0 peers';
            }

            // Knowledge Base Status
            const kbStatus = document.getElementById('kb-status');
            const kbEntries = document.getElementById('kb-entries');

            if (status.kb_active) {
                kbStatus.textContent = '📚 Activo';
                kbStatus.className = 'text-lg font-bold component-active';
                kbEntries.textContent = 'Cargando...';
            } else {
                kbStatus.textContent = '📚 Inactivo';
                kbStatus.className = 'text-lg font-bold component-inactive';
                kbEntries.textContent = '0 entradas';
            }

            // Heartbeat Status
            const heartbeatStatus = document.getElementById('heartbeat-status');
            const heartbeatHealth = document.getElementById('heartbeat-health');

            if (status.heartbeat_active) {
                heartbeatStatus.textContent = '💓 Activo';
                heartbeatStatus.className = 'text-lg font-bold component-active';
                heartbeatHealth.textContent = 'Cargando...';
            } else {
                heartbeatStatus.textContent = '💓 Inactivo';
                heartbeatStatus.className = 'text-lg font-bold component-inactive';
                heartbeatHealth.textContent = '0% salud';
            }

            // Crypto Status
            const cryptoStatus = document.getElementById('crypto-status');
            const cryptoChannels = document.getElementById('crypto-channels');

            if (status.crypto_active) {
                cryptoStatus.textContent = '🔐 Activo';
                cryptoStatus.className = 'text-lg font-bold component-active';
                cryptoChannels.textContent = 'Cargando...';
            } else {
                cryptoStatus.textContent = '🔐 Inactivo';
                cryptoStatus.className = 'text-lg font-bold component-inactive';
                cryptoChannels.textContent = '0 canales';
            }
        }

        function updateMetrics(metrics) {
            // P2P Metrics
            if (metrics.p2p) {
                document.getElementById('p2p-connected-peers').textContent = metrics.p2p.connected_peers;
                document.getElementById('p2p-discovered-peers').textContent = metrics.p2p.discovered_peers || 0;
                document.getElementById('p2p-network-active').textContent = metrics.p2p.network_active ? 'Activo' : 'Inactivo';
                document.getElementById('p2p-peers').textContent = `${metrics.p2p.connected_peers} peers`;
            }

            // Knowledge Base Metrics
            if (metrics.knowledge_base) {
                document.getElementById('kb-total-entries').textContent = metrics.knowledge_base.total_entries;
                document.getElementById('kb-total-branches').textContent = metrics.knowledge_base.total_branches;
                document.getElementById('kb-sync-peers').textContent = metrics.knowledge_base.sync_peers;
                document.getElementById('kb-entries').textContent = `${metrics.knowledge_base.total_entries} entradas`;
            }

            // Heartbeat Metrics
            if (metrics.heartbeat) {
                document.getElementById('heartbeat-healthy-nodes').textContent = metrics.heartbeat.healthy_nodes;
                document.getElementById('heartbeat-total-nodes').textContent = metrics.heartbeat.total_nodes;
                document.getElementById('heartbeat-overall-health').textContent = `${(metrics.heartbeat.overall_health * 100).toFixed(1)}%`;
                document.getElementById('heartbeat-recovery-progress').textContent = metrics.heartbeat.recovery_in_progress;
                document.getElementById('heartbeat-health').textContent = `${(metrics.heartbeat.overall_health * 100).toFixed(1)}% salud`;
            }

            // Crypto Metrics
            if (metrics.crypto) {
                document.getElementById('crypto-security-level').textContent = metrics.crypto.security_level.toUpperCase();
                document.getElementById('crypto-active-channels').textContent = metrics.crypto.active_channels;
                document.getElementById('crypto-known-peers').textContent = metrics.crypto.known_peers;
                document.getElementById('crypto-status-indicator').textContent = 'Activo';
                document.getElementById('crypto-channels').textContent = `${metrics.crypto.active_channels} canales`;
            }
        }

        function updateRealTimeMetrics() {
            socket.emit('get_real_time_metrics');
        }

        // Update metrics every 5 seconds
        setInterval(updateRealTimeMetrics, 5000);

        // Initial load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Integrated dashboard loaded');
        });
    </script>
</body>
</html>
'''


def create_integrated_dashboard(host: str = "localhost", port: int = 5000) -> IntegratedDashboardServer:
    """Crea dashboard integrado con componentes reales"""
    return IntegratedDashboardServer(host, port)


def start_integrated_dashboard(config: Dict[str, Any]):
    """Inicia dashboard integrado con configuración"""
    host = config.get("host", "127.0.0.1")
    port = config.get("dashboard_port", 8080)

    dashboard = create_integrated_dashboard(host, port)

    # Crear directorio de templates
    import os
    os.makedirs('templates', exist_ok=True)

    # Escribir template HTML
    with open('templates/integrated_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(INTEGRATED_DASHBOARD_HTML)

    # Inicializar componentes si están disponibles
    if config.get("enable_components", True):
        asyncio.run(dashboard.initialize_components(config))

    logger.info(f"🌐 Dashboard integrado iniciado en http://{host}:{port}")
    logger.info("📊 Dashboard con métricas reales de P2P, Knowledge Base, Heartbeat y Crypto")

    return dashboard


if __name__ == "__main__":
    # Configuración para testing
    config = {
        "node_id": "dashboard_integration_test",
        "enable_p2p": True,
        "enable_knowledge_base": True,
        "enable_heartbeat": True,
        "enable_crypto": True,
        "p2p": {
            "node_id": "dashboard_p2p_test",
            "port": 8080,
            "heartbeat_interval_sec": 30
        },
        "knowledge_base": {
            "node_id": "dashboard_kb_test",
            "storage_path": None
        },
        "heartbeat": {
            "node_id": "dashboard_heartbeat_test",
            "heartbeat_interval_sec": 30,
            "heartbeat_timeout_sec": 10
        },
        "crypto": {
            "node_id": "dashboard_crypto_test",
            "security_level": "HIGH"
        }
    }

    dashboard = start_integrated_dashboard(config)
    dashboard.start_server()
