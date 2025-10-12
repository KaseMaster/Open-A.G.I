import sys
import os
import asyncio
import time
from typing import List

# Asegurar que el directorio raíz del proyecto esté en sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from p2p_network import P2PNetworkManager, PeerInfo, NodeType, NetworkProtocol, ConnectionStatus, MessageType


async def run_test():
    node1_id = "smoke_node_1"
    node2_id = "smoke_node_2"
    port1 = 8091
    port2 = 8092

    # Crear dos nodos
    n1 = P2PNetworkManager(node_id=node1_id, node_type=NodeType.FULL, port=port1)
    n2 = P2PNetworkManager(node_id=node2_id, node_type=NodeType.FULL, port=port2)

    # Iniciar ambas redes
    asyncio.create_task(n1.start_network())
    asyncio.create_task(n2.start_network())

    # Esperar inicialización
    await asyncio.sleep(3)

    # Construir PeerInfo para conectar n1 -> n2 directamente
    peer2 = PeerInfo(
        peer_id=node2_id,
        node_type=NodeType.FULL,
        ip_address="127.0.0.1",
        port=port2,
        public_key="",
        capabilities=[],
        last_seen=time.time(),
        connection_status=ConnectionStatus.DISCONNECTED,
        reputation_score=0.0,
        latency=0.0,
        bandwidth=0,
        supported_protocols=[NetworkProtocol.TCP],
    )

    ok = await n1.connection_manager.connect_to_peer(peer2)
    print(f"connect_to_peer result: {ok}")
    await asyncio.sleep(2)

    # Enviar mensaje seguro desde n1 a n2
    sent = await n1.send_message(peer_id=node2_id, message_type=MessageType.DATA, payload={"hello": "from n1"})
    print(f"n1->n2 send_message: {sent}")

    # Broadcast desde n2
    bcount = await n2.broadcast_message(message_type=MessageType.BROADCAST, payload={"broadcast": "from n2"})
    print(f"n2 broadcast sent to {bcount} peers")

    # Dejar correr unos segundos para ver logs de recepción
    await asyncio.sleep(10)

    # Detener redes
    await n1.stop_network()
    await n2.stop_network()


if __name__ == "__main__":
    asyncio.run(run_test())