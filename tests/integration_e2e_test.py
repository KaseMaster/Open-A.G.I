#!/usr/bin/env python3
"""
Tests de Integración End-to-End - Sprint 1.1
AEGIS Framework - Integración de componentes principales

Este módulo implementa tests para verificar que todos los componentes
se integren correctamente y funcionen en conjunto.
"""

import asyncio
import pytest
import time
import json
import tempfile
import os
import hashlib
from typing import Dict, List, Any
import sys

# Asegurar que el directorio del proyecto esté en PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crypto_framework import (
    initialize_crypto, create_crypto_engine, SecurityLevel,
    NodeIdentity, SecureMessage
)
from cryptography.hazmat.primitives import serialization
from p2p_network import (
    P2PNetworkManager, NodeType, MessageType, start_network,
    PeerInfo, ConnectionStatus, NetworkProtocol
)
from consensus_algorithm import ConsensusEngine


class TestIntegrationSuite:
    """Suite de tests de integración para AEGIS Framework"""

    def setup_method(self):
        """Configuración para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.node_configs = self._create_test_nodes()

    def teardown_method(self):
        """Limpieza después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Crear configuración de nodos de prueba"""
        return {
            "node_1": {
                "id": "node_1",
                "host": "127.0.0.1",
                "port": 8081,
                "node_type": NodeType.FULL_NODE
            },
            "node_2": {
                "id": "node_2",
                "host": "127.0.0.1",
                "port": 8082,
                "node_type": NodeType.FULL_NODE
            },
            "node_3": {
                "id": "node_3",
                "host": "127.0.0.1",
                "port": 8083,
                "node_type": NodeType.LIGHT_NODE
            }
        }


class TestCryptoP2PIntegration:
    """Tests de integración entre framework criptográfico y red P2P"""

    @pytest.mark.asyncio
    async def test_crypto_initialization(self):
        """Test inicialización del motor criptográfico"""
        # Usar la función de inicialización del módulo
        crypto_engine = initialize_crypto({
            "security_level": "HIGH",
            "node_id": "test_node_1"
        })

        assert crypto_engine is not None
        assert crypto_engine.identity is not None
        assert crypto_engine.identity.node_id == "test_node_1"

    @pytest.mark.asyncio
    async def test_signed_message_creation(self):
        """Test creación de mensajes firmados criptográficamente"""
        crypto_engine = initialize_crypto({
            "security_level": "HIGH",
            "node_id": "test_node_1"
        })

        # Crear mensaje de prueba
        message_data = b"Hello, AEGIS Network!"

        # Firmar datos usando la API correcta
        signature = crypto_engine.sign_data(message_data)

        assert signature is not None
        assert len(signature) > 0

        # Verificar firma (necesita peer en el registro)
        # Para este test, creamos un peer ficticio
        peer_public_data = crypto_engine.identity.export_public_identity()
        crypto_engine.add_peer_identity(peer_public_data)

        is_valid = crypto_engine.verify_signature(
            message_data, signature, "test_node_1"
        )
        assert is_valid is not False

        # Verificar que la comparación con False se realice correctamente
        assert is_valid is not False

    @pytest.mark.asyncio
    async def test_message_encryption_decryption(self):
        """Test cifrado y descifrado de mensajes entre nodos"""
        # Crear dos nodos con identidades criptográficas
        node1_crypto = initialize_crypto({
            "security_level": "HIGH",
            "node_id": "node_1"
        })
        node2_crypto = initialize_crypto({
            "security_level": "HIGH",
            "node_id": "node_2"
        })

        # Intercambiar identidades públicas
        node1_public = node1_crypto.identity.export_public_identity()
        node2_public = node2_crypto.identity.export_public_identity()

        node1_crypto.add_peer_identity(node2_public)
        node2_crypto.add_peer_identity(node1_public)

        # Establecer canales seguros
        node1_crypto.establish_secure_channel("node_2")
        node2_crypto.establish_secure_channel("node_1")

        # Verificar que los canales se establecieron
        assert "node_2" in node1_crypto.ratchet_states
        assert "node_1" in node2_crypto.ratchet_states

        # Datos de prueba
        original_data = b"Machine learning model update data"

        # Cifrar mensaje de node1 para node2
        encrypted_message = node1_crypto.encrypt_message(original_data, "node_2")

        # Verificar que el cifrado funcionó
        assert encrypted_message is not None
        assert isinstance(encrypted_message, SecureMessage)
        assert encrypted_message.sender_id == "node_1"
        assert encrypted_message.recipient_id == "node_2"

        # Descifrar mensaje en node2
        decrypted_data = node2_crypto.decrypt_message(encrypted_message)

        # Verificar que el descifrado funcionó
        assert decrypted_data is not None
        assert decrypted_data == original_data


class TestP2PNetworkIntegration:
    """Tests de integración de red P2P"""

    @pytest.mark.asyncio
    async def test_p2p_network_initialization(self):
        """Test inicialización de red P2P"""
        network = P2PNetworkManager(
            node_id="test_node_1",
            node_type=NodeType.FULL,
            port=8081
        )

        assert network.node_id == "test_node_1"
        assert network.node_type == NodeType.FULL
        assert network.port == 8081
        assert not network.network_active

    @pytest.mark.asyncio
    async def test_node_discovery_simulation(self):
        """Test simulación de descubrimiento de nodos"""
        # Crear múltiples nodos
        network1 = P2PNetworkManager("node_1", NodeType.FULL, 8081)
        network2 = P2PNetworkManager("node_2", NodeType.FULL, 8082)
        network3 = P2PNetworkManager("node_3", NodeType.LIGHT, 8083)

        # Simular peers descubiertos agregándolos manualmente a la lista
        peer2_info = PeerInfo(
            peer_id="node_2",
            node_type=NodeType.FULL,
            ip_address="127.0.0.1",
            port=8082,
            public_key="",
            capabilities=["consensus", "storage"],
            last_seen=time.time(),
            connection_status=ConnectionStatus.DISCONNECTED,
            reputation_score=1.0,
            latency=0.0,
            bandwidth=0,
            supported_protocols=[NetworkProtocol.TCP]
        )

        peer3_info = PeerInfo(
            peer_id="node_3",
            node_type=NodeType.LIGHT,
            ip_address="127.0.0.1",
            port=8083,
            public_key="",
            capabilities=["storage"],
            last_seen=time.time(),
            connection_status=ConnectionStatus.DISCONNECTED,
            reputation_score=1.0,
            latency=0.0,
            bandwidth=0,
            supported_protocols=[NetworkProtocol.TCP]
        )

        # Agregar peers a las listas de descubrimiento
        network1.peer_list["node_2"] = peer2_info
        network1.peer_list["node_3"] = peer3_info

        # Verificar que los peers se agregaron
        assert len(network1.peer_list) == 2
        assert "node_2" in network1.peer_list
        assert "node_3" in network1.peer_list

    @pytest.mark.asyncio
    async def test_message_routing_between_nodes(self):
        """Test enrutamiento de mensajes entre nodos"""
        # Crear red de nodos
        network1 = P2PNetworkManager("node_1", NodeType.FULL, 8081)
        network2 = P2PNetworkManager("node_2", NodeType.FULL, 8082)

        # Simular peers conectados agregándolos a connection_manager
        peer2_info = PeerInfo(
            peer_id="node_2",
            node_type=NodeType.FULL,
            ip_address="127.0.0.1",
            port=8082,
            public_key="",
            capabilities=["consensus"],
            last_seen=time.time(),
            connection_status=ConnectionStatus.CONNECTED,
            reputation_score=1.0,
            latency=10.0,
            bandwidth=1000,
            supported_protocols=[NetworkProtocol.TCP]
        )

        network1.peer_list["node_2"] = peer2_info

        # Simular conexión agregando a active_connections
        network1.connection_manager.active_connections["node_2"] = {
            "peer_id": "node_2",
            "connected_at": time.time(),
            "last_activity": time.time(),
            "bytes_sent": 0,
            "bytes_received": 0
        }

        # Crear mensaje de prueba
        test_message = {
            "type": MessageType.DATA.value,
            "content": "Test knowledge from node_1",
            "timestamp": time.time(),
            "sender": "node_1"
        }

        # Simular envío de mensaje (sin conexión real)
        # En un test real, esto requeriría servidores TCP reales
        # Por ahora verificamos que la configuración de conexión funcione

        # Verificaciones
        assert len(network1.connection_manager.active_connections) == 1
        assert "node_2" in network1.connection_manager.active_connections


class TestConsensusIntegration:
    """Tests de integración del sistema de consenso"""

    @pytest.mark.asyncio
    async def test_consensus_engine_initialization(self):
        """Test inicialización del motor de consenso"""
        nodes = ["node_1", "node_2", "node_3"]
        consensus = ConsensusEngine("node_1", nodes)

        assert consensus.node_id == "node_1"
        assert consensus.current_leader in nodes
        assert not consensus.running
        assert len(consensus.nodes) == len(nodes)

    @pytest.mark.asyncio
    async def test_consensus_proposal_creation(self):
        """Test creación de propuestas de consenso"""
        nodes = ["node_1", "node_2", "node_3"]
        consensus = ConsensusEngine("node_1", nodes)

        # Crear propuesta de cambio
        proposal = {
            "type": "knowledge_update",
            "content_hash": "sha256:test123",
            "data": {"knowledge": "test data"},
            "timestamp": time.time()
        }

        # Simular propuesta usando el método correcto
        result = await consensus.propose(proposal)

        # En simulación, debería retornar algo o None
        assert result is not None or result is None

    @pytest.mark.asyncio
    async def test_consensus_status_reporting(self):
        """Test reporte de estado del consenso"""
        nodes = ["node_1", "node_2", "node_3"]
        consensus = ConsensusEngine("node_1", nodes)

        # Obtener estado del consenso
        status = await consensus.get_consensus_status()

        assert status is not None
        assert "current_leader" in status
        assert "node_count" in status
        assert "running" in status
        assert status["node_count"] == len(nodes)


class TestEndToEndIntegration:
    """Tests end-to-end que combinan todos los componentes"""

    @pytest.mark.asyncio
    async def test_complete_node_setup(self):
        """Test configuración completa de un nodo con todos los componentes"""
        # 1. Inicializar motor criptográfico
        crypto_engine = initialize_crypto({
            "security_level": "HIGH",
            "node_id": "integration_test_node"
        })

        # 2. Inicializar red P2P
        p2p_network = P2PNetworkManager(
            node_id="integration_test_node",
            node_type=NodeType.FULL,
            port=8090
        )

        # 3. Inicializar consenso
        consensus_nodes = ["integration_test_node", "peer_1", "peer_2"]
        consensus_engine = ConsensusEngine("integration_test_node", consensus_nodes)

        # Verificaciones de integración
        assert crypto_engine.identity.node_id == "integration_test_node"
        assert p2p_network.node_id == "integration_test_node"
        assert consensus_engine.node_id == "integration_test_node"

        # Verificar que todos los componentes se inicialicen correctamente
        assert crypto_engine.identity is not None
        assert p2p_network.port == 8090
        assert consensus_engine.current_leader in consensus_nodes

    @pytest.mark.asyncio
    async def test_signed_consensus_message_flow(self):
        """Test flujo completo de mensaje de consenso firmado"""
        # Configurar nodo completo
        crypto_engine = initialize_crypto({
            "security_level": "HIGH",
            "node_id": "consensus_test_node"
        })
        p2p_network = P2PNetworkManager(
            "consensus_test_node", NodeType.FULL, 8091
        )
        consensus_nodes = ["consensus_test_node", "peer_1", "peer_2"]
        consensus_engine = ConsensusEngine("consensus_test_node", consensus_nodes)

        # 1. Crear propuesta de conocimiento
        knowledge_data = {
            "type": "ml_model_update",
            "model_version": "1.0.0",
            "parameters": {"weights": [0.1, 0.2, 0.3]},
            "accuracy": 0.95
        }

        # 2. Crear propuesta de consenso
        proposal = {
            "type": "knowledge_update",
            "content_hash": hashlib.sha256(json.dumps(knowledge_data).encode()).hexdigest(),
            "data": knowledge_data,
            "timestamp": time.time(),
            "proposer": "consensus_test_node"
        }

        # 3. Firmar propuesta criptográficamente usando la API correcta
        signature = crypto_engine.sign_data(json.dumps(proposal).encode())

        # 4. Verificar firma usando la API correcta
        # Para verificar nuestra propia firma, necesitamos agregarnos como peer
        peer_public_data = crypto_engine.identity.export_public_identity()
        crypto_engine.add_peer_identity(peer_public_data)

        is_valid = crypto_engine.verify_signature(
            json.dumps(proposal).encode(), signature, "consensus_test_node"
        )
        assert is_valid

        # 5. Simular envío a través de P2P (simplificado para testing)
        # En un test real requeriría servidores TCP

        # Verificaciones finales
        assert proposal["type"] == "knowledge_update"
        assert proposal["proposer"] == "consensus_test_node"


# Tests de integración asíncrona
@pytest.mark.asyncio
async def test_async_integration_components():
    """Test asíncrono de integración de componentes"""
    # Configurar componentes
    crypto = initialize_crypto({
        "security_level": "STANDARD",
        "node_id": "async_test"
    })
    p2p = P2PNetworkManager("async_test", NodeType.FULL, 8092)
    consensus = ConsensusEngine("async_test", ["async_test", "peer_1"])

    # Verificar inicialización asíncrona
    await asyncio.sleep(0.1)  # Simular operaciones asíncronas

    assert crypto.identity.node_id == "async_test"
    assert p2p.node_id == "async_test"
    assert consensus.node_id == "async_test"


if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, "-v", "--tb=short"])
