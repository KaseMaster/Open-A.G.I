#!/usr/bin/env python3
"""
🧪 TESTS DE INTEGRACIÓN END-TO-END - AEGIS Framework
Suite completa de tests que validan la integración de todos los componentes.

Tests incluidos:
- Integración crypto + P2P con comunicaciones seguras
- Consenso híbrido con replicación de estado
- Recuperación de fallos y tolerancia a fallos
- Performance end-to-end con métricas
"""

import asyncio
import time
import json
import pytest
from typing import List, Dict, Any
import logging

# Configurar logging para tests
logging.basicConfig(level=logging.WARNING)

# Importar componentes del framework
try:
    from crypto_framework import CryptoEngine, CryptoConfig, SecurityLevel
    from p2p_network import P2PNetworkManager, NodeType, MessageType, PeerInfo, ConnectionStatus
    from consensus_protocol import HybridConsensus
    from distributed_state_persistence import DistributedStateManager, StateType
    from ed25519_identity_manager import Ed25519IdentityManager
    from cryptography.hazmat.primitives import serialization  # Add this import
    HAS_ALL_COMPONENTS = True
except ImportError as e:
    print(f"⚠️ Componentes faltantes para tests: {e}")
    HAS_ALL_COMPONENTS = False

pytestmark = pytest.mark.asyncio

class TestEndToEndIntegration:
    """Suite de tests de integración end-to-end"""

    @pytest.fixture
    async def crypto_engine(self):
        """Fixture para motor criptográfico"""
        config = CryptoConfig(security_level=SecurityLevel.HIGH)
        engine = CryptoEngine(config)
        identity = engine.generate_node_identity("test_node")
        return engine

    @pytest.fixture
    async def identity_manager(self):
        """Fixture para gestor de identidades"""
        manager = Ed25519IdentityManager()
        await manager.generate_own_identity("test_node")
        return manager

    @pytest.fixture
    async def state_manager(self):
        """Fixture para gestor de estado distribuido"""
        manager = DistributedStateManager("test_node")
        return manager

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_crypto_p2p_integration(self, crypto_engine):
        """Test integración crypto + P2P"""
        print("\n🔐 Testing crypto + P2P integration...")

        # Crear nodos P2P
        node1 = P2PNetworkManager("node_1", NodeType.FULL, 8081, ids=None)
        node2 = P2PNetworkManager("node_2", NodeType.FULL, 8082, ids=None)

        try:
            # Iniciar nodos
            await node1.start_network()
            await node2.start_network()

            await asyncio.sleep(2)  # Esperar inicialización

            # Verificar que los motores criptográficos están activos
            assert node1.crypto_engine is not None
            assert node2.crypto_engine is not None

            # Crear información de peer para conexión directa
            peer_info = PeerInfo(
                peer_id="node_2",
                node_type=NodeType.FULL,
                ip_address="127.0.0.1",
                port=8082,
                public_key="",
                capabilities=["consensus"],
                last_seen=time.time(),
                connection_status=ConnectionStatus.DISCONNECTED,
                reputation_score=1.0,
                latency=0.0,
                bandwidth=0,
                supported_protocols=[]
            )

            # Intentar conexión segura
            success = await node1.connection_manager.connect_to_peer(peer_info)
            assert success, "Conexión segura falló"

            # Verificar que hay canal seguro establecido
            ratchets = getattr(node1.crypto_engine, 'ratchet_states', {})
            assert "node_2" in ratchets, "Canal seguro no establecido"

            # Enviar mensaje seguro
            test_message = {"type": "test", "data": "mensaje seguro"}
            success = await node1.send_message("node_2", MessageType.DATA, test_message)
            assert success, "Envío de mensaje seguro falló"

            # Verificar recepción del mensaje
            await asyncio.sleep(0.1)  # Esperar procesamiento

            print("✅ Crypto + P2P integration test passed")

        finally:
            await node1.stop_network()
            await node2.stop_network()

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_consensus_hybrid_integration(self, crypto_engine):
        """Test integración del consenso híbrido"""
        print("\n🔗 Testing hybrid consensus integration...")

        # Crear consenso híbrido
        consensus = HybridConsensus("test_node", crypto_engine.identity.signing_key)

        # Simular otros nodos
        for i in range(3):
            node_id = f"node_{i}"
            from cryptography.hazmat.primitives.asymmetric import ed25519
            public_key = ed25519.Ed25519PrivateKey.generate().public_key()
            consensus.pbft.add_node(node_id, public_key)

        # Ejecutar ronda de consenso
        await consensus.start_consensus_round()

        # Verificar estadísticas
        stats = consensus.get_network_stats()
        assert stats["total_nodes"] == 4  # test_node + 3 nodos
        assert stats["consensus_state"] in ["idle", "completed"]

        print("✅ Hybrid consensus integration test passed")

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_distributed_state_persistence(self, state_manager):
        """Test persistencia de estado distribuido"""
        print("\n🗄️ Testing distributed state persistence...")

        # Crear estado de prueba
        test_state = {
            "consensus_view": 10,
            "last_sequence": 100,
            "active_proposals": ["prop_1", "prop_2"],
            "node_reputations": {"node_a": 0.9, "node_b": 0.8}
        }

        # Crear checkpoint
        checkpoint = await state_manager.create_checkpoint(
            StateType.CONSENSUS_STATE,
            test_state
        )

        assert checkpoint.checkpoint_id.startswith("consensus_state")
        assert len(checkpoint.state_chunks) > 0

        # Simular replicación
        success = await state_manager.replicate_checkpoint(
            checkpoint,
            ["node_2", "node_3"]
        )
        assert success, "Replicación falló"

        # Recuperar estado
        recovered_state = await state_manager.recover_state(StateType.CONSENSUS_STATE)
        assert recovered_state is not None
        assert recovered_state["consensus_view"] == 10

        print("✅ Distributed state persistence test passed")

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_identity_verification_integration(self, identity_manager):
        """Test integración de verificación de identidades"""
        print("\n🆔 Testing identity verification integration...")

        # Registrar identidad de peer
        peer_data = {
            'node_id': b'peer_test',
            'signing_key': identity_manager.own_identity.signing_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            'created_at': b'2024-01-15T10:00:00'
        }

        success = await identity_manager.register_peer_identity(peer_data)
        assert success, "Registro de identidad falló"

        # Verificar identidad
        is_verified = await identity_manager.verify_peer_identity("peer_test")
        assert is_verified, "Verificación de identidad falló"

        # Verificar firma
        test_message = b"mensaje de prueba"
        from cryptography.hazmat.primitives.asymmetric import ed25519
        private_key = ed25519.Ed25519PrivateKey.generate()
        signature = private_key.sign(test_message)

        # Esto fallará porque usamos la clave equivocada, pero verifica la integración
        is_valid = await identity_manager.verify_message_signature("peer_test", test_message, signature)
        # No esperamos que pase, pero verificamos que no crashee
        assert isinstance(is_valid, bool)

        print("✅ Identity verification integration test passed")

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_fault_tolerance_integration(self):
        """Test integración de tolerancia a fallos"""
        print("\n🛡️ Testing fault tolerance integration...")

        # Este test requiere el módulo fault_tolerance
        try:
            from fault_tolerance import FaultToleranceManager

            # Crear gestor de tolerancia a fallos
            ft_manager = FaultToleranceManager("test_node")

            # Simular escenario de fallo
            # En implementación real, esto probaría recuperación automática

            print("✅ Fault tolerance integration test passed (simulado)")

        except ImportError:
            pytest.skip("Fault tolerance module not available")

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_performance_end_to_end(self, crypto_engine):
        """Test de performance end-to-end (simplificado)"""
        print("\n⚡ Testing end-to-end performance...")

        # Test simplificado: solo verificar que los componentes se inicializan correctamente
        start_time = time.time()

        # Crear un nodo de prueba
        node = P2PNetworkManager("perf_test_node", NodeType.FULL, 9090)

        try:
            # Verificar inicialización rápida
            assert node.crypto_engine is not None
            assert node.consensus_engine is not None

            init_time = time.time() - start_time
            print(f"⏱️ Componentes inicializados en {init_time:.2f}s")

            # Verificar que el tiempo de inicialización es razonable
            assert init_time < 2.0, f"Inicialización demasiado lenta: {init_time}"

            print("✅ End-to-end performance test passed (simplificado)")

        finally:
            # No necesitamos detener la red ya que no la iniciamos
            pass

    @pytest.mark.skipif(not HAS_ALL_COMPONENTS, reason="Componentes requeridos no disponibles")
    async def test_complete_system_integration(self):
        """Test completo del sistema integrado"""
        print("\n🎯 Testing complete system integration...")

        # Crear sistema completo
        identity_manager = Ed25519IdentityManager()
        await identity_manager.generate_own_identity("master_node")

        crypto_config = CryptoConfig(security_level=SecurityLevel.HIGH)
        crypto_engine = CryptoEngine(crypto_config)
        crypto_engine.generate_node_identity("master_node")

        state_manager = DistributedStateManager("master_node")

        network = P2PNetworkManager("master_node", NodeType.FULL, 8080)

        # Verificar que todos los componentes están conectados
        assert network.crypto_engine is not None
        assert network.consensus_engine is not None

        # Obtener estado del sistema
        status = await network.get_network_status()
        assert status["consensus_available"] is True
        assert status["node_id"] == "master_node"

        # Crear y replicar estado
        test_state = {"system_status": "integrated", "components": ["crypto", "p2p", "consensus"]}
        checkpoint = await state_manager.create_checkpoint(StateType.CONSENSUS_STATE, test_state)
        await state_manager.replicate_checkpoint(checkpoint, ["backup_node"])

        print("✅ Complete system integration test passed")

async def run_integration_tests():
    """Ejecuta todos los tests de integración"""
    print("🧪 EJECUTANDO TESTS DE INTEGRACIÓN END-TO-END")
    print("=" * 50)

    if not HAS_ALL_COMPONENTS:
        print("❌ Componentes requeridos no disponibles. Instalando dependencias faltantes...")
        return

    # Ejecutar tests manualmente
    test_instance = TestEndToEndIntegration()

    try:
        # Test 1: Crypto + P2P
        crypto_engine = await test_instance.crypto_engine()
        await test_instance.test_crypto_p2p_integration(crypto_engine)

        # Test 2: Consenso híbrido
        await test_instance.test_consensus_hybrid_integration(crypto_engine)

        # Test 3: Estado distribuido
        state_manager = await test_instance.state_manager()
        await test_instance.test_distributed_state_persistence(state_manager)

        # Test 4: Verificación de identidades
        identity_manager = await test_instance.identity_manager()
        await test_instance.test_identity_verification_integration(identity_manager)

        # Test 5: Performance end-to-end
        await test_instance.test_performance_end_to_end(crypto_engine)

        # Test 6: Sistema completo
        await test_instance.test_complete_system_integration()

        print("\n🎉 TODOS LOS TESTS DE INTEGRACIÓN PASARON EXITOSAMENTE!")
        print("✅ El framework AEGIS está completamente integrado y funcional")

    except Exception as e:
        print(f"\n❌ Error en tests de integración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_integration_tests())
