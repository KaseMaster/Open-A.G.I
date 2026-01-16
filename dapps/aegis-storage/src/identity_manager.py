import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from aegis_core.crypto_framework import CryptoEngine, SecurityLevel, create_crypto_engine


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StorageRole(str, Enum):
    DATA_OWNER = "DataOwner"
    DATA_AUDITOR = "DataAuditor"
    STORAGE_NODE = "StorageNode"


@dataclass(frozen=True)
class SignedStorageTx:
    payload: Dict[str, Any]
    signer_node_id: str
    signature_b64: str
    tx_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload": self.payload,
            "signer_node_id": self.signer_node_id,
            "signature_b64": self.signature_b64,
            "tx_id": self.tx_id,
        }


class StorageIdentityManager(CryptoEngine):
    def __init__(
        self,
        node_id: Optional[str] = None,
        security_level: str = "HIGH",
    ):
        level_str = str(security_level).upper()
        level = SecurityLevel[level_str] if level_str in SecurityLevel.__members__ else SecurityLevel.HIGH
        engine = create_crypto_engine(level)
        super().__init__(engine.config)
        self.generate_node_identity(node_id)

    def sign_storage_payload(self, payload: Dict[str, Any]) -> SignedStorageTx:
        if not self.identity:
            raise RuntimeError("Identidad local no inicializada")

        body = _canonical_json_bytes(payload)
        sig = self.identity.signing_key.sign(body)
        tx_id = _sha256_hex(body + sig)
        return SignedStorageTx(
            payload=payload,
            signer_node_id=self.identity.node_id,
            signature_b64=_b64encode(sig),
            tx_id=tx_id,
        )

    def verify_storage_tx(self, tx: SignedStorageTx) -> bool:
        peer = self.peer_identities.get(tx.signer_node_id)
        if peer is None:
            return False

        try:
            body = _canonical_json_bytes(tx.payload)
            sig = _b64decode(tx.signature_b64)
            peer.public_signing_key.verify(sig, body)
            expected_tx_id = _sha256_hex(body + sig)
            return expected_tx_id == tx.tx_id
        except Exception:
            return False

    def request_role(self, role: StorageRole, proof_metadata: Dict[str, Any]) -> SignedStorageTx:
        if not self.identity:
            raise RuntimeError("Identidad local no inicializada")

        payload: Dict[str, Any] = {
            "type": "aegis_storage_identity",
            "action": "request_role",
            "node_id": self.identity.node_id,
            "role": role.value,
            "proof_metadata": proof_metadata,
            "timestamp": time.time(),
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
        }
        return self.sign_storage_payload(payload)

    def approve_multisig_op(self, *, op_id: str, policy_id: str) -> SignedStorageTx:
        if not self.identity:
            raise RuntimeError("Identidad local no inicializada")

        payload: Dict[str, Any] = {
            "type": "aegis_storage_identity",
            "action": "approve_op",
            "node_id": self.identity.node_id,
            "policy_id": policy_id,
            "op_id": op_id,
            "timestamp": time.time(),
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
        }
        return self.sign_storage_payload(payload)

    def create_multisig_policy(
        self,
        operation_type: str,
        required_signers: Sequence[str],
        threshold_m: Optional[int] = None,
    ) -> SignedStorageTx:
        if not self.identity:
            raise RuntimeError("Identidad local no inicializada")

        signers = sorted(set(required_signers))
        threshold = int(threshold_m) if threshold_m is not None else max(1, len(signers))
        policy_body = _canonical_json_bytes(
            {
                "operation_type": operation_type,
                "required_signers": signers,
                "threshold_m": threshold,
            }
        )
        policy_id = _sha256_hex(policy_body)

        payload: Dict[str, Any] = {
            "type": "aegis_storage_identity",
            "action": "create_multisig_policy",
            "node_id": self.identity.node_id,
            "operation_type": operation_type,
            "required_signers": signers,
            "policy_id": policy_id,
            "threshold_m": threshold,
            "timestamp": time.time(),
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
        }
        return self.sign_storage_payload(payload)


def build_attribute_change_operation(
    *,
    target_node_id: str,
    attribute: str,
    value: bool,
    policy_id: str,
) -> Dict[str, Any]:
    op_payload = {
        "type": "aegis_storage_identity",
        "action": "attribute_change",
        "target_node_id": target_node_id,
        "attribute": attribute,
        "value": bool(value),
        "policy_id": policy_id,
        "timestamp": time.time(),
        "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
    }
    op_id = _sha256_hex(_canonical_json_bytes(op_payload))
    op_payload["op_id"] = op_id
    return op_payload


def collect_multisig_approvals(
    *,
    op_payload: Dict[str, Any],
    approvals: List[SignedStorageTx],
) -> Dict[str, Any]:
    return {
        "type": "aegis_storage_identity",
        "action": "multisig_bundle",
        "op": op_payload,
        "approvals": [a.to_dict() for a in approvals],
    }
