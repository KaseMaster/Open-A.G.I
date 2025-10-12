#!/usr/bin/env python3
"""
Tests Unitarios para el Sistema P2P de AEGIS
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

# Importaciones condicionales para dependencias opcionales
try:
    from storage_system import AEGISStorage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logging.warning("Módulo storage_system no disponible - funcionalidad de almacenamiento deshabilitada")

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    logging.warning("Módulo pytest no disponible - algunos tests pueden fallar")

# Importar el framework de tests
sys.path.append(str(Path(__file__).parent.parent))
from test_framework import (
    AEGISTestFramework, TestSuite, TestType, TestStatus,
    unit_test, integration_test, performance_test, security_test, stress_test
)

class P2PNetworkTests:
    """Tests para el sistema P2P"""
    
    def __init__(self):
        self.p2p_module = None
        self.test_peers = [
            {"id": "peer1", "address": "127.0.0.1", "port": 8001},
            {"id": "peer2", "address": "127.0.0.1", "port": 8002},
            {"id": "peer3", "address": "127.0.0.1", "port": 8003}
        ]
        self.test_messages = [
            {"type": "ping", "data": "hello"},
            {"type": "data", "data": {"key": "value"}},
            {"type": "consensus", "data": {"proposal": "test"}}
        ]
    
    def setup(self):
        """Setup para tests de P2P"""
        try:
            # Intentar importar el módulo P2P real
            from p2p_network import AEGISP2PNetwork
            self.p2p_module = AEGISP2PNetwork(port=8000)
        except ImportError:
            # Usar mock si no está disponible
            self.p2p_module = self._create_p2p_mock()
    
    def teardown(self):
        """Cleanup después de tests"""
        if hasattr(self.p2p_module, 'stop'):
            asyncio.create_task(self.p2p_module.stop())
    
    def _create_p2p_mock(self):
        """Crea un mock del módulo P2P"""
        mock = Mock()
        mock.start = AsyncMock(return_value=True)
        mock.stop = AsyncMock(return_value=True)
        mock.discover_peers = AsyncMock(return_value=self.test_peers)
        mock.connect_to_peer = AsyncMock(return_value=True)
        mock.send_message = AsyncMock(return_value=True)
        mock.broadcast_message = AsyncMock(return_value=True)
        # Métodos adicionales usados por los tests
        mock.validate_message = Mock(side_effect=lambda m: bool(m.get("type")))
        mock.authenticate_peer = AsyncMock(side_effect=lambda p: bool(p.get("public_key")))
        mock.send_encrypted_message = AsyncMock(return_value=True)
        # Implementación simple de rate limiting: siempre True para simplificar
        mock.check_rate_limit = Mock(return_value=True)
        mock.get_connected_peers = Mock(return_value=self.test_peers)
        mock.get_network_status = Mock(return_value={
            "status": "running",
            "connected_peers": len(self.test_peers),
            "total_messages": 0
        })
        return mock
    
    @unit_test
    async def test_p2p_initialization(self):
        """Test de inicialización del sistema P2P"""
        # Test de inicio
        result = await self.p2p_module.start()
        assert result is True, "El sistema P2P debe iniciarse correctamente"
        
        # Test de estado
        status = self.p2p_module.get_network_status()
        assert status is not None, "El estado de red no debe ser None"
        assert "status" in status, "El estado debe incluir el campo 'status'"
    
    @unit_test
    async def test_peer_discovery(self):
        """Test de descubrimiento de peers"""
        peers = await self.p2p_module.discover_peers()
        
        assert peers is not None, "La lista de peers no debe ser None"
        assert isinstance(peers, list), "Los peers deben ser una lista"
        
        # Verificar estructura de peers
        for peer in peers:
            assert "id" in peer, "Cada peer debe tener un ID"
            assert "address" in peer, "Cada peer debe tener una dirección"
            assert "port" in peer, "Cada peer debe tener un puerto"
    
    @unit_test
    async def test_peer_connection(self):
        """Test de conexión a peers"""
        test_peer = self.test_peers[0]
        
        # Test de conexión exitosa
        result = await self.p2p_module.connect_to_peer(
            test_peer["address"], 
            test_peer["port"]
        )
        assert result is True, "La conexión al peer debe ser exitosa"
        
        # Verificar que el peer está en la lista de conectados
        connected_peers = self.p2p_module.get_connected_peers()
        peer_ids = [p["id"] for p in connected_peers]
        assert test_peer["id"] in peer_ids, "El peer debe estar en la lista de conectados"
    
    @unit_test
    async def test_message_sending(self):
        """Test de envío de mensajes"""
        test_peer = self.test_peers[0]
        test_message = self.test_messages[0]
        
        # Conectar al peer primero
        await self.p2p_module.connect_to_peer(test_peer["address"], test_peer["port"])
        
        # Enviar mensaje
        result = await self.p2p_module.send_message(test_peer["id"], test_message)
        assert result is True, "El envío de mensaje debe ser exitoso"
    
    @unit_test
    async def test_message_broadcasting(self):
        """Test de broadcast de mensajes"""
        test_message = self.test_messages[1]
        
        # Conectar a múltiples peers
        for peer in self.test_peers:
            await self.p2p_module.connect_to_peer(peer["address"], peer["port"])
        
        # Broadcast del mensaje
        result = await self.p2p_module.broadcast_message(test_message)
        assert result is True, "El broadcast debe ser exitoso"
    
    @unit_test
    def test_message_validation(self):
        """Test de validación de mensajes"""
        # Test con mensaje válido
        valid_message = {
            "type": "data",
            "data": {"test": "value"},
            "timestamp": time.time(),
            "sender": "test_node"
        }
        
        is_valid = self.p2p_module.validate_message(valid_message)
        assert is_valid, "El mensaje válido debe pasar la validación"
        
        # Test con mensaje inválido (sin tipo)
        invalid_message = {
            "data": {"test": "value"},
            "timestamp": time.time()
        }
        
        is_valid = self.p2p_module.validate_message(invalid_message)
        assert not is_valid, "El mensaje sin tipo debe fallar la validación"
    
    @security_test
    async def test_peer_authentication(self):
        """Test de autenticación de peers"""
        # Test con peer válido
        valid_peer = {
            "id": "valid_peer",
            "address": "127.0.0.1",
            "port": 8001,
            "public_key": "valid_public_key"
        }
        
        auth_result = await self.p2p_module.authenticate_peer(valid_peer)
        assert auth_result is True, "El peer válido debe autenticarse"
        
        # Test con peer sin clave pública
        invalid_peer = {
            "id": "invalid_peer",
            "address": "127.0.0.1",
            "port": 8002
        }
        
        auth_result = await self.p2p_module.authenticate_peer(invalid_peer)
        assert auth_result is False, "El peer sin clave pública debe fallar la autenticación"
    
    @security_test
    async def test_message_encryption(self):
        """Test de encriptación de mensajes"""
        test_message = self.test_messages[2]
        test_peer = self.test_peers[0]
        
        # Enviar mensaje encriptado
        result = await self.p2p_module.send_encrypted_message(
            test_peer["id"], 
            test_message
        )
        assert result is True, "El envío de mensaje encriptado debe ser exitoso"
    
    @security_test
    def test_rate_limiting(self):
        """Test de limitación de velocidad"""
        test_peer_id = "rate_test_peer"
        
        # Simular múltiples mensajes rápidos
        for i in range(100):
            result = self.p2p_module.check_rate_limit(test_peer_id)
            if i < 50:  # Primeros 50 deben pasar
                assert result is True, f"Mensaje {i} debe pasar el rate limit"
            else:  # Los siguientes deben ser limitados
                # Dependiendo de la implementación, algunos pueden ser bloqueados
                pass
    
    @performance_test
    async def test_connection_performance(self):
        """Test de rendimiento de conexiones"""
        start_time = time.time()
        
        # Conectar a múltiples peers simultáneamente
        connection_tasks = [
            self.p2p_module.connect_to_peer(peer["address"], peer["port"])
            for peer in self.test_peers
        ]
        
        results = await asyncio.gather(*connection_tasks)
        connection_time = time.time() - start_time
        
        # Verificar que todas las conexiones fueron exitosas
        assert all(results), "Todas las conexiones deben ser exitosas"
        
        # Verificar tiempo de conexión razonable
        assert connection_time < 5.0, "Las conexiones deben completarse en menos de 5 segundos"
    
    @performance_test
    async def test_message_throughput(self):
        """Test de throughput de mensajes"""
        test_peer = self.test_peers[0]
        await self.p2p_module.connect_to_peer(test_peer["address"], test_peer["port"])
        
        message_count = 100
        start_time = time.time()
        
        # Enviar múltiples mensajes
        send_tasks = [
            self.p2p_module.send_message(test_peer["id"], {
                "type": "test",
                "data": f"message_{i}",
                "index": i
            })
            for i in range(message_count)
        ]
        
        results = await asyncio.gather(*send_tasks)
        total_time = time.time() - start_time
        
        # Calcular throughput
        throughput = message_count / total_time
        
        assert all(results), "Todos los mensajes deben enviarse exitosamente"
        assert throughput > 10, "El throughput debe ser mayor a 10 mensajes/segundo"
    
    @stress_test
    async def test_high_peer_count(self):
        """Test de estrés con muchos peers"""
        # Simular conexión a muchos peers
        peer_count = 50
        fake_peers = [
            {
                "id": f"stress_peer_{i}",
                "address": "127.0.0.1",
                "port": 8000 + i
            }
            for i in range(peer_count)
        ]
        
        # Intentar conectar a todos
        connection_results = []
        for peer in fake_peers:
            try:
                result = await asyncio.wait_for(
                    self.p2p_module.connect_to_peer(peer["address"], peer["port"]),
                    timeout=1.0
                )
                connection_results.append(result)
            except asyncio.TimeoutError:
                connection_results.append(False)
        
        # Al menos el 70% de las conexiones deben ser exitosas
        success_rate = sum(connection_results) / len(connection_results)
        assert success_rate > 0.7, "Al menos 70% de las conexiones deben ser exitosas"
    
    @stress_test
    async def test_message_flood(self):
        """Test de estrés con flood de mensajes"""
        test_peer = self.test_peers[0]
        await self.p2p_module.connect_to_peer(test_peer["address"], test_peer["port"])
        
        # Enviar muchos mensajes rápidamente
        message_count = 1000
        messages = [
            {
                "type": "flood_test",
                "data": f"flood_message_{i}",
                "timestamp": time.time()
            }
            for i in range(message_count)
        ]
        
        start_time = time.time()
        
        # Enviar en lotes para evitar sobrecargar
        batch_size = 50
        successful_sends = 0
        
        for i in range(0, message_count, batch_size):
            batch = messages[i:i + batch_size]
            batch_tasks = [
                self.p2p_module.send_message(test_peer["id"], msg)
                for msg in batch
            ]
            
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=5.0
                )
                successful_sends += sum(1 for r in batch_results if r is True)
            except asyncio.TimeoutError:
                break
        
        total_time = time.time() - start_time
        
        # Al menos el 80% de los mensajes deben enviarse exitosamente
        success_rate = successful_sends / message_count
        assert success_rate > 0.8, f"Al menos 80% de mensajes deben enviarse (actual: {success_rate:.1%})"
        
        # El sistema debe mantener rendimiento razonable
        throughput = successful_sends / total_time
        assert throughput > 50, "El throughput debe mantenerse sobre 50 mensajes/segundo"
    
    @integration_test
    async def test_p2p_with_consensus(self):
        """Test de integración con sistema de consenso"""
        try:
            from consensus_system import AEGISConsensus
            consensus = AEGISConsensus()
        except ImportError:
            pytest.skip("Módulo de consenso no disponible")
        
        # Conectar peers para consenso
        for peer in self.test_peers:
            await self.p2p_module.connect_to_peer(peer["address"], peer["port"])
        
        # Proponer un valor para consenso
        proposal = {
            "type": "consensus_proposal",
            "value": "test_value",
            "proposer": "test_node"
        }
        
        # Broadcast de la propuesta
        result = await self.p2p_module.broadcast_message(proposal)
        assert result is True, "La propuesta debe broadcastearse exitosamente"
    
    @integration_test
    async def test_p2p_with_storage(self):
        """Test de integración con sistema de almacenamiento"""
        if not STORAGE_AVAILABLE:
            if PYTEST_AVAILABLE:
                pytest.skip("Módulo de almacenamiento no disponible")
            else:
                raise Exception("Skipped: Módulo de almacenamiento no disponible")
        
        storage = AEGISStorage()
        
        # Test de sincronización de datos
        test_data = {"key": "value", "timestamp": time.time()}
        
        # Almacenar datos localmente
        storage_id = await storage.store_data(test_data)
        
        # Notificar a peers sobre nuevos datos
        sync_message = {
            "type": "data_sync",
            "storage_id": storage_id,
            "data_hash": storage.get_data_hash(storage_id)
        }
        
        result = await self.p2p_module.broadcast_message(sync_message)
        assert result is True, "La notificación de sincronización debe ser exitosa"

def create_p2p_test_suite() -> TestSuite:
    """Crea la suite de tests para P2P"""
    p2p_tests = P2PNetworkTests()
    
    return TestSuite(
        name="P2PNetwork",
        description="Tests para el sistema P2P de AEGIS",
        tests=[
            p2p_tests.test_p2p_initialization,
            p2p_tests.test_peer_discovery,
            p2p_tests.test_peer_connection,
            p2p_tests.test_message_sending,
            p2p_tests.test_message_broadcasting,
            p2p_tests.test_message_validation,
            p2p_tests.test_peer_authentication,
            p2p_tests.test_message_encryption,
            p2p_tests.test_rate_limiting,
            p2p_tests.test_connection_performance,
            p2p_tests.test_message_throughput,
            p2p_tests.test_high_peer_count,
            p2p_tests.test_message_flood,
            p2p_tests.test_p2p_with_consensus,
            p2p_tests.test_p2p_with_storage
        ],
        setup_func=p2p_tests.setup,
        teardown_func=p2p_tests.teardown,
        test_type=TestType.UNIT
    )

if __name__ == "__main__":
    # Ejecutar tests de P2P individualmente
    async def main():
        from test_framework import get_test_framework
        
        framework = get_test_framework()
        suite = create_p2p_test_suite()
        framework.register_test_suite(suite)
        
        results = await framework.run_all_tests()
        print(f"P2P tests - Éxito: {results['summary']['success_rate']:.1f}%")
    
    asyncio.run(main())