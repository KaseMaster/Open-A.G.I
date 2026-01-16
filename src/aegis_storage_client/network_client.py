import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from aegis_core.crypto_framework import CryptoEngine, SecureMessage
from aegis_core.tor_integration import TorGateway


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@dataclass(frozen=True)
class StorageNodeEndpoint:
    node_id: str
    onion: str
    port: int


class AegisStorageNetworkClient:
    def __init__(
        self,
        *,
        crypto_engine: CryptoEngine,
        tor_gateway: TorGateway,
        security_level: str = "PARANOID",
    ):
        self.crypto = crypto_engine
        self.tor = tor_gateway
        self.security_level = str(security_level).upper()

    async def initialize(self) -> bool:
        return await self.tor.initialize()

    async def upload_fragments(
        self,
        *,
        target: StorageNodeEndpoint,
        file_id: str,
        fragments: Dict[int, bytes],
        merkle_root_hex: str,
        fragment_hashes_hex: Dict[int, str],
    ) -> bool:
        for idx, frag_blob in fragments.items():
            payload = {
                "type": "aegis_storage",
                "action": "upload_fragment",
                "file_id": file_id,
                "fragment_index": int(idx),
                "fragment_hash": fragment_hashes_hex[int(idx)],
                "merkle_root": merkle_root_hex,
                "fragment_blob_b64": _b64(frag_blob),
            }
            ok = await self._send_secure_json(target, payload)
            if not ok:
                return False
        return True

    async def initiate_access_request(self, *, target: StorageNodeEndpoint, file_id: str) -> bool:
        payload = {
            "type": "aegis_storage",
            "action": "access_request",
            "file_id": file_id,
        }
        return await self._send_secure_json(target, payload)

    async def challenge_node_integrity(
        self,
        *,
        target: StorageNodeEndpoint,
        file_id: str,
        fragment_index: int,
        fragment_hash_hex: str,
        challenge_nonce_b64: str,
    ) -> bool:
        payload = {
            "type": "aegis_storage",
            "action": "integrity_challenge",
            "file_id": file_id,
            "fragment_index": int(fragment_index),
            "fragment_hash": fragment_hash_hex,
            "nonce_b64": challenge_nonce_b64,
        }
        return await self._send_secure_json(target, payload)

    async def _send_secure_json(self, target: StorageNodeEndpoint, payload: Dict[str, Any]) -> bool:
        if not self.crypto.identity:
            raise RuntimeError("crypto identity missing")
        if target.node_id not in self.crypto.peer_identities:
            raise RuntimeError("peer identity missing")
        if target.node_id not in self.crypto.ratchet_states:
            ok = self.crypto.establish_secure_channel(target.node_id)
            if not ok:
                raise RuntimeError("secure channel not established")

        plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sm = self.crypto.encrypt_message(plaintext, target.node_id)
        if sm is None:
            return False

        envelope = {
            "type": "secure",
            "sender_id": sm.sender_id,
            "recipient_id": sm.recipient_id,
            "secure_message_b64": _b64(sm.serialize()),
        }
        body = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return await self.tor.send_message(target.onion, target.port, body)


def decode_secure_envelope(envelope_bytes: bytes) -> Dict[str, Any]:
    msg = json.loads(envelope_bytes.decode("utf-8"))
    if msg.get("type") != "secure":
        raise ValueError("not secure")
    return msg


def extract_secure_message(envelope: Dict[str, Any]) -> SecureMessage:
    sm_b64 = envelope.get("secure_message_b64")
    if not isinstance(sm_b64, str):
        raise ValueError("missing secure_message_b64")
    raw = base64.b64decode(sm_b64.encode("ascii"))
    return SecureMessage.deserialize(raw)

