#!/usr/bin/env python3
"""
Tests para Sistema de Heartbeat Distribuido - AEGIS Framework
Pruebas de funcionalidad del sistema de heartbeat con recuperación automática.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock

# Asegurar que el directorio del proyecto esté en PYTHONPATH
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distributed_heartbeat import (
    DistributedHeartbeatManager, HeartbeatStatus, RecoveryStrategy,
    HeartbeatMetrics, NetworkPath, create_heartbeat_manager
)


class MockP2PManager:
    """P2P Manager mock para testing"""

    def __init__(self):
        self.sent_messages = []
        self.connected_peers = ["node_1", "node_2", "node_3"]

    async def send_message(self, target_node: str, message_type, payload: dict) -> bool:
        """Mock envío de mensaje"""
        self.sent_messages.append({
            "target": target_node,
            "type": message_type,
            "payload": payload,
            "timestamp": time.time()
        })

        # Simular éxito para algunos nodos
        if target_node in ["node_1", "node_2"]:
            return True
        else:
            return False

    async def broadcast_message(self, message_type, payload: dict) -> int:
        """Mock broadcast de mensaje"""
        count = 0
        for peer in self.connected_peers:
            success = await self.send_message(peer, message_type, payload)
            if success:
                count += 1
        return count

    async def get_network_status(self) -> dict:
        """Mock estado de red"""
        return {
            "connected_peers": self.connected_peers,
            "network_active": True,
            "topology": {"total_nodes": len(self.connected_peers)}
        }


class TestHeartbeatManager:
    """Tests del gestor de heartbeat"""

    def setup_method(self):
        """Configuración para cada test"""
        self.mock_p2p = MockP2PManager()
        self.heartbeat_manager = create_heartbeat_manager("test_node", self.mock_p2p)

    def test_heartbeat_manager_initialization(self):
        """Test inicialización del gestor de heartbeat"""
        assert self.heartbeat_manager.node_id == "test_node"
        assert self.heartbeat_manager.running == False
        assert self.heartbeat_manager.heartbeat_interval == 30
        assert self.heartbeat_manager.heartbeat_timeout == 10
        assert len(self.heartbeat_manager.node_metrics) == 0

    def test_add_remove_nodes(self):
        """Test agregar y remover nodos del sistema"""
        # Agregar nodos
        self.heartbeat_manager.add_node("node_1", 0.05)
        self.heartbeat_manager.add_node("node_2", 0.10)
        self.heartbeat_manager.add_node("node_3", 0.15)

        assert len(self.heartbeat_manager.node_metrics) == 3
        assert "node_1" in self.heartbeat_manager.node_metrics
        assert "node_2" in self.heartbeat_manager.node_metrics
        assert "node_3" in self.heartbeat_manager.node_metrics

        # Verificar métricas iniciales
        metrics1 = self.heartbeat_manager.node_metrics["node_1"]
        assert metrics1.node_id == "node_1"
        assert metrics1.get_average_response_time() == 0.05
        assert metrics1.status == HeartbeatStatus.HEALTHY

        # Remover nodo
        self.heartbeat_manager.remove_node("node_2")
        assert len(self.heartbeat_manager.node_metrics) == 2
        assert "node_2" not in self.heartbeat_manager.node_metrics

    @pytest.mark.asyncio
    async def test_heartbeat_success(self):
        """Test heartbeat exitoso"""
        self.heartbeat_manager.add_node("node_1")

        result = await self.heartbeat_manager.send_heartbeat("node_1")

        assert result["success"] == True
        assert "response_time" in result
        assert result["node_status"] == "healthy"

        # Verificar métricas actualizadas
        metrics = self.heartbeat_manager.node_metrics["node_1"]
        assert metrics.successful_heartbeats == 1
        assert metrics.total_heartbeats == 1
        assert metrics.get_success_rate() == 1.0
        assert metrics.status == HeartbeatStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_failure(self):
        """Test heartbeat fallido"""
        self.heartbeat_manager.add_node("node_3")  # Este nodo falla en el mock

        result = await self.heartbeat_manager.send_heartbeat("node_3")

        assert result["success"] == False
        assert "error" in result
        assert result["node_status"] == "degraded"

        # Verificar métricas actualizadas
        metrics = self.heartbeat_manager.node_metrics["node_3"]
        assert metrics.successful_heartbeats == 0
        assert metrics.total_heartbeats == 1
        assert metrics.consecutive_failures == 1
        assert metrics.get_success_rate() == 0.0
        assert metrics.status == HeartbeatStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_multiple_heartbeats(self):
        """Test múltiples heartbeats a diferentes nodos"""
        nodes = ["node_1", "node_2", "node_3"]
        for node_id in nodes:
            self.heartbeat_manager.add_node(node_id)

        # Enviar heartbeats a todos los nodos
        results = []
        for node_id in nodes:
            result = await self.heartbeat_manager.send_heartbeat(node_id)
            results.append(result)

        # Verificar resultados
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        assert len(successful) == 2  # node_1 y node_2 deberían tener éxito
        assert len(failed) == 1      # node_3 debería fallar

        # Verificar métricas
        status = self.heartbeat_manager.get_heartbeat_status()
        assert status["total_nodes"] == 3
        assert status["healthy_nodes"] == 2  # node_1 y node_2 están saludables
        assert status["overall_health"] == 2/3

    def test_heartbeat_metrics_calculation(self):
        """Test cálculo de métricas de heartbeat"""
        metrics = HeartbeatMetrics("test_node")

        # Simular respuestas exitosas
        metrics.record_success(0.05)
        metrics.record_success(0.10)
        metrics.record_success(0.08)

        assert metrics.get_success_rate() == 1.0
        assert abs(metrics.get_average_response_time() - 0.076) < 0.01
        assert metrics.consecutive_failures == 0
        assert metrics.is_healthy() == True

        # Simular fallos
        metrics.record_failure()
        metrics.record_failure()

        assert metrics.get_success_rate() == 3/5  # 3 exitosos de 5 totales
        assert metrics.consecutive_failures == 2
        assert metrics.status == HeartbeatStatus.DEGRADED
        assert metrics.needs_recovery() == False  # Solo 2 fallos, necesita 3+ para UNRESPONSIVE

    def test_network_paths_management(self):
        """Test gestión de rutas de red"""
        # Crear rutas de prueba
        path1 = NetworkPath("path_1", ["node_A", "node_B"])
        path2 = NetworkPath("path_2", ["node_A", "node_C", "node_B"])

        # Verificar propiedades iniciales
        assert path1.path_id == "path_1"
        assert len(path1.nodes) == 2
        assert path1.get_success_rate() == 1.0  # Sin intentos aún

        # Registrar algunos intentos
        path1.record_attempt(True, 0.05)
        path1.record_attempt(True, 0.08)
        path1.record_attempt(False, 0.0)

        assert path1.total_attempts == 3
        assert path1.successful_attempts == 2
        assert path1.get_success_rate() == 2/3

    @pytest.mark.asyncio
    async def test_recovery_strategies(self):
        """Test estrategias de recuperación"""
        # Configurar nodo con fallos
        self.heartbeat_manager.add_node("failing_node")
        metrics = self.heartbeat_manager.node_metrics["failing_node"]

        # Simular múltiples fallos para activar recuperación
        for _ in range(3):
            metrics.record_failure()

        assert metrics.status == HeartbeatStatus.UNRESPONSIVE
        assert metrics.needs_recovery() == True

        # Ejecutar recuperación (simulada)
        await self.heartbeat_manager._execute_recovery_strategy(
            "failing_node", RecoveryStrategy.RETRY, metrics
        )

        # Verificar que se actualizaron las métricas de recuperación
        assert metrics.recovery_attempts == 1
        assert metrics.last_recovery_attempt > 0


class TestHeartbeatIntegration:
    """Tests de integración del sistema de heartbeat"""

    @pytest.mark.asyncio
    async def test_heartbeat_system_lifecycle(self):
        """Test ciclo de vida completo del sistema de heartbeat"""
        mock_p2p = MockP2PManager()
        heartbeat_manager = create_heartbeat_manager("integration_node", mock_p2p)

        # Agregar nodos
        nodes = ["peer_1", "peer_2", "peer_3"]
        for node_id in nodes:
            heartbeat_manager.add_node(node_id)

        # Iniciar sistema
        await heartbeat_manager.start_heartbeat_system()

        assert heartbeat_manager.running == True

        # Esperar un ciclo de heartbeat
        await asyncio.sleep(0.5)

        # Verificar que se enviaron heartbeats
        assert len(mock_p2p.sent_messages) > 0

        # Obtener estado del sistema
        status = heartbeat_manager.get_heartbeat_status()
        assert status["total_nodes"] == 3
        assert "overall_health" in status

        # Detener sistema
        await heartbeat_manager.stop_heartbeat_system()

        assert heartbeat_manager.running == False

    @pytest.mark.asyncio
    async def test_heartbeat_with_p2p_integration(self):
        """Test heartbeat integrado con P2P manager"""
        # Este test verifica la integración real con el P2P manager
        # Nota: requeriría un P2P manager real para funcionar completamente

        mock_p2p = MockP2PManager()
        heartbeat_manager = create_heartbeat_manager("p2p_node", mock_p2p)

        # Agregar nodos conectados
        for peer in mock_p2p.connected_peers:
            heartbeat_manager.add_node(peer)

        # Enviar heartbeats
        for peer in mock_p2p.connected_peers:
            result = await heartbeat_manager.send_heartbeat(peer)

            # node_1 y node_2 deberían tener éxito, node_3 debería fallar
            if peer in ["node_1", "node_2"]:
                assert result["success"] == True
            else:
                assert result["success"] == False

        # Verificar estado final
        status = heartbeat_manager.get_heartbeat_status()
        assert status["total_nodes"] == 3
        assert status["healthy_nodes"] == 2  # node_1 y node_2


# Tests de métricas y monitoreo
class TestHeartbeatMetrics:
    """Tests de métricas y monitoreo de heartbeat"""

    def test_metrics_calculations(self):
        """Test cálculos de métricas"""
        metrics = HeartbeatMetrics("metrics_test")

        # Simular patrón de respuestas
        response_times = [0.01, 0.02, 0.015, 0.03, 0.025]

        for rt in response_times:
            metrics.record_success(rt)

        assert metrics.get_success_rate() == 1.0
        assert metrics.get_average_response_time() == 0.02  # Promedio de 0.02
        assert metrics.total_heartbeats == 5
        assert metrics.successful_heartbeats == 5
        assert metrics.is_healthy() == True

    def test_failure_detection(self):
        """Test detección de fallos"""
        metrics = HeartbeatMetrics("failure_test")

        # Simular fallos progresivos
        for i in range(6):
            if i < 3:
                metrics.record_success(0.05)
            else:
                metrics.record_failure()

        # Verificar detección de estado
        assert metrics.status == HeartbeatStatus.FAILED  # 3 fallos consecutivos
        assert metrics.consecutive_failures == 3
        assert metrics.get_success_rate() == 3/6  # 50%
        assert metrics.needs_recovery() == True

    def test_health_transitions(self):
        """Test transiciones de estado de salud"""
        metrics = HeartbeatMetrics("transition_test")

        # Estado inicial: saludable
        assert metrics.status == HeartbeatStatus.HEALTHY
        assert metrics.is_healthy() == True
        assert metrics.needs_recovery() == False

        # 1 fallo: degradado
        metrics.record_failure()
        assert metrics.status == HeartbeatStatus.DEGRADED
        assert metrics.is_healthy() == True  # Aún se considera saludable

        # 3 fallos: no responsivo
        metrics.record_failure()
        metrics.record_failure()
        assert metrics.status == HeartbeatStatus.UNRESPONSIVE
        assert metrics.needs_recovery() == True

        # Recuperación
        metrics.record_success(0.05)
        assert metrics.status == HeartbeatStatus.HEALTHY
        assert metrics.consecutive_failures == 0


# Tests asíncronos avanzados
@pytest.mark.asyncio
async def test_async_heartbeat_operations():
    """Test operaciones asíncronas del sistema de heartbeat"""
    mock_p2p = MockP2PManager()
    heartbeat_manager = create_heartbeat_manager("async_node", mock_p2p)

    # Configurar nodos
    nodes = ["async_peer_1", "async_peer_2"]
    for node_id in nodes:
        heartbeat_manager.add_node(node_id)

    # Ejecutar operaciones asíncronas
    await asyncio.sleep(0.1)

    # Verificar estado asíncrono
    status = heartbeat_manager.get_heartbeat_status()
    assert status["total_nodes"] == 2
    assert status["overall_health"] == 1.0  # Todos saludables inicialmente

    # Simular heartbeats asíncronos
    heartbeat_tasks = []
    for node_id in nodes:
        task = heartbeat_manager.send_heartbeat(node_id)
        heartbeat_tasks.append(task)

    results = await asyncio.gather(*heartbeat_tasks)

    # Verificar resultados
    successful_results = [r for r in results if r["success"]]
    assert len(successful_results) == 2  # Ambos deberían tener éxito

    # Verificar estado final
    final_status = heartbeat_manager.get_heartbeat_status()
    assert final_status["total_nodes"] == 2
    assert final_status["healthy_nodes"] == 2


if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, "-v", "--tb=short"])
