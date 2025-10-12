import asyncio
import time
import base64
import os
import sys
from typing import Any, Dict, List

# Asegurar que el directorio del proyecto está en sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cryptography.hazmat.primitives.asymmetric import ed25519

from consensus_protocol import PBFTConsensus, ConsensusMessage, MessageType


class DummyNetworkManager:
    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    async def broadcast_message(self, channel_type, payload: Dict[str, Any]):
        # Solo capturamos el payload para inspección
        self.sent_messages.append({"channel_type": channel_type, "payload": payload})


async def _async_outgoing_signature_added_and_valid():
    # Identidad local
    priv = ed25519.Ed25519PrivateKey.generate()
    node_id = "node_local"
    nm = DummyNetworkManager()
    pbft = PBFTConsensus(node_id=node_id, private_key=priv, network_manager=nm)

    # Crear mensaje y transmitir
    msg = ConsensusMessage(
        message_type=MessageType.PROPOSAL,
        sender_id=node_id,
        view_number=0,
        sequence_number=1,
        payload={"foo": "bar"},
        timestamp=time.time(),
        signature=None,
    )
    await pbft._broadcast_message(msg)

    assert nm.sent_messages, "No se capturó ningún mensaje transmitido"
    payload = nm.sent_messages[-1]["payload"]
    assert isinstance(payload.get("signature"), str), "La firma no fue incluida en el payload"

    # Verificar firma con la clave pública local
    sig_b64 = payload["signature"]
    signature_bytes = base64.b64decode(sig_b64)
    # Reconstruir dict canónico
    canonical = {
        "message_type": payload["message_type"],
        "sender_id": payload["sender_id"],
        "view_number": payload["view_number"],
        "sequence_number": payload["sequence_number"],
        "payload": payload["payload"],
        "timestamp": payload["timestamp"],
    }
    msg_bytes = __import__("json").dumps(canonical, sort_keys=True).encode()
    pub = priv.public_key()
    pub.verify(signature_bytes, msg_bytes)


async def _async_incoming_signature_verification():
    # Setup PBFT y handler
    priv_local = ed25519.Ed25519PrivateKey.generate()
    node_id = "node_local"
    pbft = PBFTConsensus(node_id=node_id, private_key=priv_local, network_manager=None)

    received: List[ConsensusMessage] = []

    async def handler(cm: ConsensusMessage):
        received.append(cm)

    pbft.message_handlers[MessageType.PREPARE] = handler

    # Nodo remoto
    priv_remote = ed25519.Ed25519PrivateKey.generate()
    remote_id = "peer_A"
    pbft.add_node(remote_id, priv_remote.public_key())

    # Construir payload firmado válido
    payload_valid = {
        "message_type": MessageType.PREPARE.value,
        "sender_id": remote_id,
        "view_number": 0,
        "sequence_number": 2,
        "payload": {"x": 1},
        "timestamp": time.time(),
    }
    canonical_bytes = __import__("json").dumps(payload_valid, sort_keys=True).encode()
    sig = priv_remote.sign(canonical_bytes)
    payload_valid["signature"] = base64.b64encode(sig).decode()

    # Entregar mensaje válido
    await pbft._on_consensus_network_message(remote_id, {"payload": payload_valid})
    assert received and received[-1].sender_id == remote_id, "El mensaje válido no fue despachado"

    # Construir payload con firma inválida (alterar un byte)
    bad_sig = bytearray(sig)
    bad_sig[0] ^= 0xFF
    payload_invalid = dict(payload_valid)
    payload_invalid["signature"] = base64.b64encode(bytes(bad_sig)).decode()

    # Entregar mensaje inválido
    before = len(received)
    await pbft._on_consensus_network_message(remote_id, {"payload": payload_invalid})
    assert len(received) == before, "El mensaje con firma inválida no debe ser despachado"


def test_outgoing_signature_added_and_valid():
    """Pytest-compatible sync wrapper for the async outgoing signature test."""
    asyncio.run(_async_outgoing_signature_added_and_valid())


def test_incoming_signature_verification():
    """Pytest-compatible sync wrapper for the async incoming signature verification test."""
    asyncio.run(_async_incoming_signature_verification())


def run():
    asyncio.run(test_outgoing_signature_added_and_valid())
    asyncio.run(test_incoming_signature_verification())


if __name__ == "__main__":
    run()