from typing import Any, Dict, Optional

from aegis_core.consensus_protocol import PBFTConsensus

from identity_manager import StorageIdentityManager
from storage_identity_state import StorageIdentityState


def initialize_dapp(
    *,
    crypto_engine: Optional[Any],
    p2p_manager: Optional[Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    security_level = config.get("security_level", "HIGH")
    node_id = config.get("node_id")

    identity_manager = StorageIdentityManager(node_id=node_id, security_level=security_level)
    if crypto_engine is not None and getattr(crypto_engine, "identity", None) is not None:
        identity_manager.identity = crypto_engine.identity
        identity_manager.peer_identities = getattr(crypto_engine, "peer_identities", {})

    pbft = None
    if identity_manager.identity is not None:
        pbft = PBFTConsensus(
            node_id=identity_manager.identity.node_id,
            private_key=identity_manager.identity.signing_key,
            network_manager=p2p_manager,
        )

    state = StorageIdentityState()
    return {
        "identity_manager": identity_manager,
        "pbft": pbft,
        "state": state,
    }
