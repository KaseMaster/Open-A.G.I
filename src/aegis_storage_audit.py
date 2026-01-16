import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from aegis_core.crypto_framework import CryptoEngine


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def merkle_root_from_hashes(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        return b""
    level = list(leaves)
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else a
            nxt.append(_sha256(a + b))
        level = nxt
    return level[0]


def merkle_proof_from_hashes(leaves: Sequence[bytes], index: int) -> List[Tuple[bytes, str]]:
    if index < 0 or index >= len(leaves):
        raise IndexError
    level = list(leaves)
    idx = index
    proof: List[Tuple[bytes, str]] = []
    while len(level) > 1:
        if idx % 2 == 0:
            sib_idx = idx + 1
            pos = "right"
        else:
            sib_idx = idx - 1
            pos = "left"
        sibling = level[sib_idx] if sib_idx < len(level) else level[idx]
        proof.append((sibling, pos))
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else a
            nxt.append(_sha256(a + b))
        level = nxt
        idx //= 2
    return proof


def merkle_verify_hash(leaf_hash: bytes, proof: Sequence[Tuple[bytes, str]], root: bytes) -> bool:
    h = leaf_hash
    for sib, pos in proof:
        h = _sha256(sib + h) if pos == "left" else _sha256(h + sib)
    return h == root


def merkle_compute_root(leaf_hash: bytes, proof: Sequence[Tuple[bytes, str]]) -> bytes:
    h = leaf_hash
    for sib, pos in proof:
        h = _sha256(sib + h) if pos == "left" else _sha256(h + sib)
    return h


@dataclass(frozen=True)
class AuditEvent:
    id: str
    timestamp: float
    actor_id: str
    event_type: str
    status: str
    payload: Dict[str, Any]
    signature_b64: str

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "event_type": self.event_type,
            "status": self.status,
            "payload": self.payload,
            "signature_b64": self.signature_b64,
        }

    def leaf_hash(self) -> bytes:
        return _sha256(_canonical_json_bytes(self.canonical_dict()))


class MerkleAuditLog:
    def __init__(self, *, persist_path: Optional[str] = None):
        self._events: List[AuditEvent] = []
        self._leaf_hashes: List[bytes] = []
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def count(self) -> int:
        return len(self._events)

    def current_root(self) -> str:
        root = merkle_root_from_hashes(self._leaf_hashes)
        return root.hex()

    def append(self, ev: AuditEvent) -> int:
        idx = len(self._events)
        self._events.append(ev)
        self._leaf_hashes.append(ev.leaf_hash())
        if self.persist_path:
            with self.persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev.canonical_dict(), ensure_ascii=False) + "\n")
        return idx

    def get_event(self, idx: int) -> AuditEvent:
        return self._events[idx]

    def list_events(self, *, query: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        q = (query or "").lower().strip()
        out: List[Dict[str, Any]] = []
        for i in range(len(self._events) - 1, -1, -1):
            ev = self._events[i]
            leaf = self._leaf_hashes[i].hex()
            if q and q not in ev.id.lower() and q not in leaf and q not in ev.event_type.lower():
                continue
            out.append(
                {
                    "id": ev.id,
                    "timestamp": ev.timestamp,
                    "event_type": ev.event_type,
                    "status": ev.status,
                    "actor_id": ev.actor_id,
                    "index": i,
                    "leaf_hash": leaf,
                }
            )
            if len(out) >= int(limit):
                break
        return out

    def proof(self, idx: int) -> Dict[str, Any]:
        ev = self._events[idx]
        leaf_hash = self._leaf_hashes[idx]
        proof = merkle_proof_from_hashes(self._leaf_hashes, idx)
        siblings = [{"hash": h.hex(), "position": pos} for (h, pos) in proof]
        return {
            "leaf_hash": leaf_hash.hex(),
            "index": idx,
            "siblings": siblings,
            "root": self.current_root(),
            "event": ev.canonical_dict(),
        }

    def verify_proof(self, *, leaf_hash_hex: str, index: int, siblings: Sequence[Dict[str, Any]], root_hex: str) -> bool:
        leaf = bytes.fromhex(leaf_hash_hex)
        root = bytes.fromhex(root_hex)
        proof: List[Tuple[bytes, str]] = []
        for s in siblings:
            proof.append((bytes.fromhex(str(s["hash"])), str(s["position"])))
        return merkle_verify_hash(leaf, proof, root)

    def compute_root_from_request(self, *, leaf_hash_hex: str, siblings: Sequence[Dict[str, Any]]) -> str:
        leaf = bytes.fromhex(leaf_hash_hex)
        proof: List[Tuple[bytes, str]] = []
        for s in siblings:
            proof.append((bytes.fromhex(str(s["hash"])), str(s["position"])))
        return merkle_compute_root(leaf, proof).hex()

    def verify_event_signature(self, ev: AuditEvent, *, public_signing_key: Ed25519PublicKey) -> bool:
        try:
            msg = _canonical_json_bytes(
                {
                    "id": ev.id,
                    "timestamp": ev.timestamp,
                    "actor_id": ev.actor_id,
                    "event_type": ev.event_type,
                    "status": ev.status,
                    "payload": ev.payload,
                }
            )
            public_signing_key.verify(_b64d(ev.signature_b64), msg)
            return True
        except Exception:
            return False


def create_audit_event(
    *,
    crypto: CryptoEngine,
    event_type: str,
    status: str,
    payload: Dict[str, Any],
    event_id: Optional[str] = None,
    ts: Optional[float] = None,
) -> AuditEvent:
    if not crypto.identity:
        raise RuntimeError("crypto identity missing")
    now = float(ts if ts is not None else time.time())
    eid = event_id or hashlib.sha256(os.urandom(32)).hexdigest()
    actor = crypto.identity.node_id
    msg = _canonical_json_bytes(
        {
            "id": eid,
            "timestamp": now,
            "actor_id": actor,
            "event_type": str(event_type),
            "status": str(status),
            "payload": payload,
        }
    )
    sig = crypto.sign_data(msg)
    return AuditEvent(
        id=eid,
        timestamp=now,
        actor_id=actor,
        event_type=str(event_type),
        status=str(status),
        payload=payload,
        signature_b64=_b64(sig),
    )

