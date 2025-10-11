#!/usr/bin/env python3
"""
Pruebas de integración de componentes clave (Crypto, Consenso, P2P).
Evitan procesos persistentes y verifican inicializaciones ligeras.
"""

import os
import sys
import asyncio

# Asegurar que el directorio del proyecto esté en PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_crypto_initialize_engine():
    from crypto_framework import initialize_crypto
    engine = initialize_crypto({"security_level": "HIGH", "node_id": "testnode"})
    assert engine is not None
    assert engine.identity is not None
    assert engine.identity.node_id == "testnode"


def test_consensus_engine_init_and_leader():
    from consensus_algorithm import ConsensusEngine

    nodes = ["node_a", "node_b", "node_c"]
    engine = ConsensusEngine("node_a", nodes)

    # Validaciones básicas sin iniciar el servicio
    assert engine.current_leader in nodes
    assert engine.running is False

    status = asyncio.run(engine.get_consensus_status())
    assert "current_leader" in status
    assert status["node_count"] == len(nodes)


def test_hybrid_consensus_stats():
    from consensus_protocol import HybridConsensus
    from cryptography.hazmat.primitives.asymmetric import ed25519

    priv = ed25519.Ed25519PrivateKey.generate()
    hc = HybridConsensus("hc_node", priv)

    # Añadir algunos nodos simulados
    for i in range(3):
        other_key = ed25519.Ed25519PrivateKey.generate()
        hc.pbft.add_node(f"node_{i}", other_key.public_key())

    stats = hc.get_network_stats()
    assert "total_nodes" in stats
    assert "consensus_state" in stats
    assert stats["total_nodes"] >= 1


def test_p2p_enums_available():
    import p2p_network as p2p
    # Verificar que las enumeraciones principales existen
    assert hasattr(p2p, "NodeType")
    assert hasattr(p2p, "MessageType")