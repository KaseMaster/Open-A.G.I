#!/usr/bin/env python3
"""
Sistema de Heartbeat Distribuido - AEGIS Framework
Implementación de heartbeat multi-path con recuperación automática de fallos.

Características:
- Heartbeat distribuido con múltiples rutas
- Detección proactiva de fallos de nodos
- Recuperación automática con reintentos inteligentes
- Métricas de latencia y disponibilidad
- Integración con sistema de reputación

Programador Principal: Jose Gómez alias KaseMaster
Contacto: kasemaster@aegis-framework.com
Versión: 2.0.0
Licencia: MIT
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
from datetime import datetime, timezone

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HeartbeatStatus(Enum):
    """Estados de heartbeat"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"


class RecoveryStrategy(Enum):
    """Estrategias de recuperación"""
    RETRY = "retry"
    REROUTE = "reroute"
    ISOLATE = "isolate"
    REPLACE = "replace"


@dataclass
class HeartbeatMetrics:
    """Métricas de heartbeat para un nodo"""
    node_id: str
    last_heartbeat: float = 0.0
    response_times: List[float] = field(default_factory=list)
    consecutive_failures: int = 0
    total_heartbeats: int = 0
    successful_heartbeats: int = 0
    status: HeartbeatStatus = HeartbeatStatus.HEALTHY
    last_recovery_attempt: float = 0.0
    recovery_attempts: int = 0

    def add_response_time(self, response_time: float):
        """Agrega tiempo de respuesta y mantiene historial"""
        self.response_times.append(response_time)
        self.total_heartbeats += 1

        # Mantener solo los últimos 100 tiempos de respuesta
        if len(self.response_times) > 100:
            self.response_times.pop(0)

    def record_success(self, response_time: float):
        """Registra heartbeat exitoso"""
        self.add_response_time(response_time)
        self.successful_heartbeats += 1
        self.consecutive_failures = 0
        self.status = HeartbeatStatus.HEALTHY
        self.last_heartbeat = time.time()

    def record_failure(self):
        """Registra heartbeat fallido"""
        self.consecutive_failures += 1
        self.total_heartbeats += 1

        # Determinar estado basado en fallos consecutivos
        if self.consecutive_failures >= 5:
            self.status = HeartbeatStatus.FAILED
        elif self.consecutive_failures >= 3:
            self.status = HeartbeatStatus.UNRESPONSIVE
        elif self.consecutive_failures >= 1:
            self.status = HeartbeatStatus.DEGRADED

    def get_success_rate(self) -> float:
        """Calcula tasa de éxito de heartbeats"""
        if self.total_heartbeats == 0:
            return 1.0
        return self.successful_heartbeats / self.total_heartbeats

    def get_average_response_time(self) -> float:
        """Calcula tiempo de respuesta promedio"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    def is_healthy(self) -> bool:
        """Determina si el nodo está en estado saludable"""
        return self.status == HeartbeatStatus.HEALTHY

    def needs_recovery(self) -> bool:
        """Determina si el nodo necesita recuperación"""
        return self.status in [HeartbeatStatus.UNRESPONSIVE, HeartbeatStatus.FAILED]


@dataclass
class NetworkPath:
    """Ruta de red para heartbeat multi-path"""
    path_id: str
    nodes: List[str]  # Secuencia de nodos en la ruta
    latency: float = 0.0
    reliability: float = 1.0
    last_used: float = 0.0
    total_attempts: int = 0
    successful_attempts: int = 0

    def get_success_rate(self) -> float:
        """Calcula tasa de éxito de la ruta"""
        if self.total_attempts == 0:
            return 1.0
        return self.successful_attempts / self.total_attempts

    def record_attempt(self, success: bool, latency: float = 0.0):
        """Registra intento de uso de la ruta"""
        self.total_attempts += 1
        self.last_used = time.time()

        if success:
            self.successful_attempts += 1
            self.latency = latency
            self.reliability = self.get_success_rate()
        else:
            self.reliability = self.get_success_rate()


class DistributedHeartbeatManager:
    """Gestor de heartbeat distribuido principal"""

    def __init__(self, node_id: str, p2p_manager: Any = None):
        self.node_id = node_id
        self.p2p_manager = p2p_manager

        # Estado de heartbeat
        self.node_metrics: Dict[str, HeartbeatMetrics] = {}
        self.network_paths: Dict[str, NetworkPath] = {}
        self.recovery_callbacks: List[Callable] = []

        # Configuración
        self.heartbeat_interval = 30  # segundos
        self.heartbeat_timeout = 10  # segundos
        self.max_consecutive_failures = 5
        self.recovery_retry_interval = 60  # segundos
        self.path_discovery_interval = 300  # 5 minutos

        # Estado del sistema
        self.running = False
        self.recovery_in_progress: Set[str] = set()

        logger.info(f"💓 Heartbeat Manager inicializado para nodo {node_id}")

    def add_node(self, node_id: str, initial_latency: float = 0.0):
        """Agrega nodo al sistema de heartbeat"""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = HeartbeatMetrics(
                node_id=node_id,
                last_heartbeat=time.time()
            )

            # Agregar latencia inicial si se proporciona
            if initial_latency > 0:
                self.node_metrics[node_id].add_response_time(initial_latency)

            logger.debug(f"➕ Nodo agregado al heartbeat: {node_id}")

    def remove_node(self, node_id: str):
        """Remueve nodo del sistema de heartbeat"""
        if node_id in self.node_metrics:
            del self.node_metrics[node_id]
            logger.debug(f"➖ Nodo removido del heartbeat: {node_id}")

    def discover_network_paths(self):
        """Descubre rutas alternativas para heartbeat multi-path"""
        if not self.p2p_manager:
            return

        try:
            # Obtener topología de red actual
            network_status = asyncio.run(self.p2p_manager.get_network_status())
            connected_peers = network_status.get("connected_peers", [])

            # Crear rutas directas para cada peer
            for peer_id in connected_peers:
                if peer_id != self.node_id:
                    path_id = f"direct_{self.node_id}_{peer_id}"
                    self.network_paths[path_id] = NetworkPath(
                        path_id=path_id,
                        nodes=[self.node_id, peer_id]
                    )

            # Crear rutas multi-hop si hay suficientes nodos
            if len(connected_peers) >= 3:
                self._create_multi_hop_paths(connected_peers)

            logger.info(f"🛤️ Descubiertas {len(self.network_paths)} rutas de heartbeat")

        except Exception as e:
            logger.error(f"❌ Error descubriendo rutas de red: {e}")

    def _create_multi_hop_paths(self, peers: List[str]):
        """Crea rutas multi-hop para mayor resiliencia"""
        # Implementación simplificada: crear rutas a través de nodos intermedios
        for i, peer1 in enumerate(peers):
            for j, peer2 in enumerate(peers):
                if i != j and peer1 != self.node_id and peer2 != self.node_id:
                    # Crear ruta a través de peer1
                    path_id = f"via_{peer1}_{self.node_id}_{peer2}"
                    self.network_paths[path_id] = NetworkPath(
                        path_id=path_id,
                        nodes=[self.node_id, peer1, peer2]
                    )

    async def send_heartbeat(self, target_node: str) -> Dict[str, Any]:
        """Envía heartbeat a nodo específico"""
        if target_node not in self.node_metrics:
            self.add_node(target_node)

        metrics = self.node_metrics[target_node]
        start_time = time.time()

        try:
            # Crear mensaje de heartbeat
            heartbeat_msg = {
                "type": "heartbeat",
                "sender": self.node_id,
                "target": target_node,
                "timestamp": start_time,
                "sequence": metrics.total_heartbeats + 1,
                "metadata": {
                    "response_expected": True,
                    "timeout": self.heartbeat_timeout
                }
            }

            # Enviar a través de múltiples rutas si están disponibles
            if self.network_paths:
                await self._send_heartbeat_multi_path(target_node, heartbeat_msg)
            else:
                await self._send_heartbeat_direct(target_node, heartbeat_msg)

            # Calcular tiempo de respuesta
            response_time = time.time() - start_time
            metrics.record_success(response_time)

            return {
                "success": True,
                "response_time": response_time,
                "node_status": metrics.status.value
            }

        except Exception as e:
            logger.warning(f"⚠️ Heartbeat fallido a {target_node}: {e}")
            metrics.record_failure()

            return {
                "success": False,
                "error": str(e),
                "node_status": metrics.status.value
            }

    async def _send_heartbeat_direct(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Envía heartbeat directamente al nodo"""
        if not self.p2p_manager:
            return False

        try:
            # Usar el sistema de P2P para enviar heartbeat
            success = await self.p2p_manager.send_message(
                target_node,
                MessageType.HEARTBEAT,
                message
            )
            return success

        except Exception as e:
            logger.debug(f"❌ Error en heartbeat directo a {target_node}: {e}")
            return False

    async def _send_heartbeat_multi_path(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Envía heartbeat usando múltiples rutas"""
        if not self.p2p_manager:
            return await self._send_heartbeat_direct(target_node, message)

        # Encontrar rutas disponibles hacia el nodo objetivo
        available_paths = [
            path for path in self.network_paths.values()
            if target_node in path.nodes
        ]

        if not available_paths:
            return await self._send_heartbeat_direct(target_node, message)

        # Usar la ruta más confiable
        best_path = max(available_paths, key=lambda p: p.reliability)
        best_path.record_attempt(False)  # Marcar intento

        try:
            # Enviar a través de la mejor ruta
            if len(best_path.nodes) == 2:  # Ruta directa
                success = await self._send_heartbeat_direct(target_node, message)
            else:  # Ruta multi-hop
                success = await self._send_heartbeat_via_intermediate(
                    best_path.nodes[1], target_node, message
                )

            best_path.record_attempt(success)
            return success

        except Exception as e:
            logger.debug(f"❌ Error en heartbeat multi-path a {target_node}: {e}")
            best_path.record_attempt(False)
            return False

    async def _send_heartbeat_via_intermediate(self, intermediate_node: str,
                                             target_node: str, message: Dict[str, Any]) -> bool:
        """Envía heartbeat a través de nodo intermedio"""
        if not self.p2p_manager:
            return False

        try:
            # Modificar mensaje para routing
            routed_message = {
                **message,
                "routing": {
                    "via": intermediate_node,
                    "final_destination": target_node
                }
            }

            # Enviar al nodo intermedio
            success = await self.p2p_manager.send_message(
                intermediate_node,
                MessageType.HEARTBEAT,
                routed_message
            )

            return success

        except Exception as e:
            logger.debug(f"❌ Error en heartbeat vía {intermediate_node}: {e}")
            return False

    async def heartbeat_loop(self):
        """Bucle principal de heartbeat"""
        logger.info(f"💓 Iniciando bucle de heartbeat para {len(self.node_metrics)} nodos")

        while self.running:
            try:
                # Enviar heartbeat a todos los nodos conocidos
                heartbeat_tasks = []
                for node_id in list(self.node_metrics.keys()):
                    if node_id != self.node_id:
                        task = asyncio.create_task(self.send_heartbeat(node_id))
                        heartbeat_tasks.append((node_id, task))

                # Esperar respuestas con timeout
                for node_id, task in heartbeat_tasks:
                    try:
                        await asyncio.wait_for(task, timeout=self.heartbeat_timeout + 5)
                    except asyncio.TimeoutError:
                        logger.warning(f"⏰ Timeout en heartbeat a {node_id}")
                        if node_id in self.node_metrics:
                            self.node_metrics[node_id].record_failure()

                # Ejecutar recuperación si es necesario
                await self._check_and_recover_failed_nodes()

                # Actualizar rutas de red periódicamente
                await self._update_network_paths()

                # Esperar hasta próximo heartbeat
                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"❌ Error en bucle de heartbeat: {e}")
                await asyncio.sleep(10)

    async def _check_and_recover_failed_nodes(self):
        """Verifica y recupera nodos fallidos"""
        for node_id, metrics in list(self.node_metrics.items()):
            if metrics.needs_recovery() and node_id not in self.recovery_in_progress:
                asyncio.create_task(self._recover_node(node_id))

    async def _recover_node(self, node_id: str):
        """Intenta recuperar un nodo fallido"""
        self.recovery_in_progress.add(node_id)
        try:
            metrics = self.node_metrics.get(node_id)
            if not metrics:
                return

            # Determinar estrategia basada en historial
            strategy = RecoveryStrategy.RETRY
            if metrics.recovery_attempts > 3:
                strategy = RecoveryStrategy.REROUTE
            if metrics.recovery_attempts > 6:
                strategy = RecoveryStrategy.ISOLATE

            await self._execute_recovery_strategy(node_id, strategy, metrics)
            metrics.recovery_attempts += 1
            metrics.last_recovery_attempt = time.time()
            
        finally:
            self.recovery_in_progress.discard(node_id)

    async def _execute_recovery_strategy(self, node_id: str, strategy: RecoveryStrategy,
                                       metrics: HeartbeatMetrics):
        """Ejecuta estrategia de recuperación específica"""
        logger.info(f"🔧 Ejecutando estrategia {strategy.value} para {node_id}")

        if strategy == RecoveryStrategy.RETRY:
            # Reintentar heartbeat con backoff exponencial
            await self._retry_with_backoff(node_id, metrics)

        elif strategy == RecoveryStrategy.REROUTE:
            # Intentar rutas alternativas
            await self._try_alternative_routes(node_id)

        elif strategy == RecoveryStrategy.ISOLATE:
            # Aislar nodo problemático
            await self._isolate_problematic_node(node_id)

        elif strategy == RecoveryStrategy.REPLACE:
            # Buscar reemplazo o reconectar
            await self._find_node_replacement(node_id)

    async def _retry_with_backoff(self, node_id: str, metrics: HeartbeatMetrics):
        """Reintenta conexión con backoff exponencial"""
        base_delay = 1
        max_retries = 3

        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)

            logger.debug(f"⏳ Reintentando {node_id} en {delay}s (intento {attempt + 1})")
            await asyncio.sleep(delay)

            # Intentar heartbeat
            result = await self.send_heartbeat(node_id)

            if result["success"]:
                logger.info(f"✅ Recuperación exitosa de {node_id}")
                return

        logger.warning(f"❌ No se pudo recuperar {node_id} después de {max_retries} intentos")

    async def _try_alternative_routes(self, node_id: str):
        """Intenta rutas alternativas para llegar al nodo"""
        if not self.p2p_manager:
            return

        # Buscar rutas alternativas
        alternative_paths = [
            path for path in self.network_paths.values()
            if node_id in path.nodes and path.get_success_rate() > 0.5
        ]

        for path in alternative_paths[:3]:  # Probar máximo 3 rutas
            logger.debug(f"🛤️ Probando ruta alternativa para {node_id}: {path.path_id}")

            # Crear heartbeat con información de ruta
            heartbeat_msg = {
                "type": "heartbeat_recovery",
                "sender": self.node_id,
                "target": node_id,
                "route": path.nodes,
                "timestamp": time.time()
            }

            try:
                if await self._send_heartbeat_via_path(node_id, path, heartbeat_msg):
                    logger.info(f"✅ Ruta alternativa exitosa para {node_id}")
                    return

            except Exception as e:
                logger.debug(f"❌ Ruta alternativa fallida: {e}")

        logger.warning(f"❌ No se encontraron rutas alternativas para {node_id}")

    async def _send_heartbeat_via_path(self, target_node: str, path: NetworkPath,
                                      message: Dict[str, Any]) -> bool:
        """Envía heartbeat a través de una ruta específica"""
        # Implementación simplificada: enviar a través del primer nodo intermedio
        if len(path.nodes) >= 2:
            intermediate = path.nodes[1]
            return await self._send_heartbeat_direct(intermediate, message)

        return False

    async def _isolate_problematic_node(self, node_id: str):
        """Aísla nodo problemático"""
        logger.warning(f"🚫 Aislando nodo problemático {node_id}")

        # Marcar como aislado
        if node_id in self.node_metrics:
            self.node_metrics[node_id].status = HeartbeatStatus.FAILED

        # Notificar a otros componentes
        await self._notify_node_isolation(node_id)

    async def _find_node_replacement(self, node_id: str):
        """Busca reemplazo para nodo fallido"""
        if not self.p2p_manager:
            return

        logger.info(f"🔄 Buscando reemplazo para nodo fallido {node_id}")

        # Obtener lista de peers disponibles
        try:
            network_status = await self.p2p_manager.get_network_status()
            available_peers = [
                peer for peer in network_status.get("peer_list", [])
                if peer.get("peer_id") != node_id and peer.get("connection_status") == "connected"
            ]

            if available_peers:
                # Seleccionar mejor candidato basado en métricas
                best_peer = max(available_peers,
                              key=lambda p: p.get("reputation_score", 0))

                logger.info(f"✅ Reemplazo encontrado: {best_peer['peer_id']}")

                # Migrar responsabilidades (implementación específica de la aplicación)
                await self._migrate_node_responsibilities(node_id, best_peer["peer_id"])

        except Exception as e:
            logger.error(f"❌ Error buscando reemplazo: {e}")

    async def _notify_node_isolation(self, node_id: str):
        """Notifica aislamiento de nodo a otros componentes"""
        notification = {
            "type": "node_isolation",
            "node_id": node_id,
            "timestamp": time.time(),
            "reason": "heartbeat_failures"
        }

        # Notificar a través del sistema de P2P
        if self.p2p_manager:
            try:
                await self.p2p_manager.broadcast_message(
                    MessageType.BROADCAST,
                    notification
                )
            except Exception as e:
                logger.error(f"❌ Error notificando aislamiento: {e}")

    async def _migrate_node_responsibilities(self, old_node: str, new_node: str):
        """Migra responsabilidades del nodo fallido al nuevo"""
        # Implementación específica de la aplicación
        logger.info(f"🔄 Migrando responsabilidades de {old_node} a {new_node}")

        # En una implementación real, esto involucraría:
        # - Transferir datos del nodo fallido
        # - Actualizar routing tables
        # - Reasignar tareas y responsabilidades
        # - Actualizar topología de red

    async def _update_network_paths(self):
        """Actualiza rutas de red disponibles"""
        try:
            # Descubrir nuevas rutas periódicamente
            self.discover_network_paths()

            # Actualizar métricas de rutas existentes
            current_time = time.time()
            for path in self.network_paths.values():
                # Reducir confiabilidad de rutas no usadas recientemente
                if current_time - path.last_used > 300:  # 5 minutos
                    path.reliability *= 0.9  # Decaimiento exponencial

        except Exception as e:
            logger.error(f"❌ Error actualizando rutas de red: {e}")

    def get_heartbeat_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema de heartbeat"""
        healthy_nodes = 0
        total_nodes = len(self.node_metrics)

        for metrics in self.node_metrics.values():
            if metrics.is_healthy():
                healthy_nodes += 1

        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "unresponsive_nodes": len([m for m in self.node_metrics.values() if m.status == HeartbeatStatus.UNRESPONSIVE]),
            "failed_nodes": len([m for m in self.node_metrics.values() if m.status == HeartbeatStatus.FAILED]),
            "overall_health": healthy_nodes / total_nodes if total_nodes > 0 else 1.0,
            "network_paths": len(self.network_paths),
            "recovery_in_progress": len(self.recovery_in_progress),
            "node_details": {
                node_id: {
                    "status": metrics.status.value,
                    "success_rate": metrics.get_success_rate(),
                    "avg_response_time": metrics.get_average_response_time(),
                    "consecutive_failures": metrics.consecutive_failures
                }
                for node_id, metrics in self.node_metrics.items()
            }
        }

    def add_recovery_callback(self, callback: Callable[[str, RecoveryStrategy], Any]):
        """Agrega callback para notificaciones de recuperación"""
        self.recovery_callbacks.append(callback)

    async def start_heartbeat_system(self):
        """Inicia el sistema de heartbeat"""
        logger.info("🚀 Iniciando sistema de heartbeat distribuido")

        self.running = True

        # Descubrir rutas iniciales
        self.discover_network_paths()

        # Iniciar bucle de heartbeat
        asyncio.create_task(self.heartbeat_loop())

        logger.info("✅ Sistema de heartbeat iniciado")

    async def stop_heartbeat_system(self):
        """Detiene el sistema de heartbeat"""
        logger.info("🛑 Deteniendo sistema de heartbeat")

        self.running = False

        # Cancelar todas las tareas de recuperación en progreso
        for node_id in list(self.recovery_in_progress):
            self.recovery_in_progress.discard(node_id)

        logger.info("✅ Sistema de heartbeat detenido")


# Funciones de integración
def create_heartbeat_manager(node_id: str, p2p_manager: Any = None) -> DistributedHeartbeatManager:
    """Crea gestor de heartbeat distribuido"""
    return DistributedHeartbeatManager(node_id, p2p_manager)


def initialize_heartbeat_system(config: Dict[str, Any]) -> DistributedHeartbeatManager:
    """Inicializa sistema de heartbeat desde configuración"""
    node_id = config.get("node_id", "node_local")
    heartbeat_interval = config.get("heartbeat_interval_sec", 30)
    heartbeat_timeout = config.get("heartbeat_timeout_sec", 10)

    manager = create_heartbeat_manager(node_id)
    manager.heartbeat_interval = heartbeat_interval
    manager.heartbeat_timeout = heartbeat_timeout

    logger.info(f"💓 Sistema de heartbeat inicializado para nodo {node_id}")
    return manager


if __name__ == "__main__":
    # Demo del sistema de heartbeat
    async def demo_heartbeat_system():
        print("💓 Demo del Sistema de Heartbeat Distribuido")
        print("=" * 50)

        # Crear gestor de heartbeat
        heartbeat_manager = create_heartbeat_manager("demo_node")

        # Agregar nodos de prueba
        test_nodes = ["node_1", "node_2", "node_3", "node_4"]
        for node_id in test_nodes:
            heartbeat_manager.add_node(node_id)

        print(f"✅ Agregados {len(test_nodes)} nodos al sistema de heartbeat")

        # Mostrar estado inicial
        status = heartbeat_manager.get_heartbeat_status()
        print("📊 Estado inicial:")
        print(f"  - Nodos totales: {status['total_nodes']}")
        print(f"  - Nodos saludables: {status['healthy_nodes']}")
        print(f"  - Salud general: {status['overall_health']:.1%}")

        # Simular algunos heartbeats exitosos
        print("\n🔄 Simulando heartbeats...")

        for node_id in test_nodes:
            result = await heartbeat_manager.send_heartbeat(node_id)

            if result["success"]:
                print(f"  ✅ Heartbeat exitoso a {node_id} ({result['response_time']:.2f}s)")
            else:
                print(f"  ❌ Heartbeat fallido a {node_id}")

        # Mostrar estado final
        print("\n📊 Estado final:")
        final_status = heartbeat_manager.get_heartbeat_status()
        for node_id, details in final_status["node_details"].items():
            print(f"  - {node_id}: {details['status']} (éxito: {details['success_rate']:.1%})")

        # Limpiar
        await heartbeat_manager.stop_heartbeat_system()

    asyncio.run(demo_heartbeat_system())
