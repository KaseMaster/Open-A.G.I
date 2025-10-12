import asyncio
import time
import os
import sys
from typing import Dict, Any, Callable, Awaitable, List

# Asegurar que el directorio del proyecto está en sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cryptography.hazmat.primitives.asymmetric import ed25519

# Importar PBFTConsensus desde el módulo del proyecto
from consensus_protocol import PBFTConsensus, MessageType as PBFTMessageType


class DummyNetworkManager:
    """Simula el registro de handlers de P2PNetworkManager y permite despachar mensajes."""

    def __init__(self):
        # Mapeo por tipo de mensaje (string) a lista de handlers
        self.handlers: Dict[str, List[Callable[[str, Dict[str, Any]], Awaitable[None]]]] = {}

    def register_handler(self, message_type, handler: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        # message_type es enum de p2p_network.MessageType, usar su valor string
        key = getattr(message_type, "value", str(message_type))
        self.handlers.setdefault(key, []).append(handler)

    async def dispatch(self, peer_id: str, mtype: str, message: Dict[str, Any]):
        for h in self.handlers.get(mtype, []):
            await h(peer_id, message)


async def run_test() -> None:
    nm = DummyNetworkManager()
    private_key = ed25519.Ed25519PrivateKey.generate()
    pbft = PBFTConsensus(node_id="node_X", private_key=private_key, network_manager=nm)

    # Interceptar handler de PROPOSAL para verificar que el bridging funciona
    captured: Dict[str, Any] = {}

    async def capture_proposal(message):
        captured["message_type"] = message.message_type
        captured["sender_id"] = message.sender_id
        captured["view_number"] = message.view_number
        captured["sequence_number"] = message.sequence_number
        captured["payload"] = message.payload

    # Reemplazar temporalmente el handler
    pbft.message_handlers[PBFTMessageType.PROPOSAL] = capture_proposal

    net_message = {
        "type": "consensus",
        "payload": {
            "message_type": PBFTMessageType.PROPOSAL.value,
            "sender_id": "peer_A",
            "view_number": 1,
            "sequence_number": 42,
            "payload": {"foo": "bar"},
            "timestamp": time.time(),
            "signature": None,
        },
    }

    await nm.dispatch("peer_A", "consensus", net_message)

    assert captured.get("message_type") == PBFTMessageType.PROPOSAL, "Tipo de mensaje no reconstruido correctamente"
    assert captured.get("sender_id") == "peer_A", "Sender no asignado correctamente"
    assert captured.get("view_number") == 1, "View number incorrecto"
    assert captured.get("sequence_number") == 42, "Sequence number incorrecto"
    assert captured.get("payload") == {"foo": "bar"}, "Payload reconstruido incorrectamente"


def test_consensus_bridge_proposal_handling():
    """Pytest-compatible wrapper that executes the async consensus bridge test.

    Uses asyncio.run to avoid relying on pytest-asyncio plugin and ensures
    the coroutine is executed within a proper event loop.
    """
    asyncio.run(run_test())