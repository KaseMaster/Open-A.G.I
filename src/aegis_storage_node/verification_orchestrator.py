import asyncio
import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from pbft_integration import InMemoryContractEventBus, InMemoryAegisStorageLedger

from aegis_storage_node.storage_service import PorChallenge, PorResponse, StorageService


class NodeTransport:
    async def request_por(self, node_id: str, challenge: PorChallenge) -> PorResponse:
        raise NotImplementedError


class InMemoryNodeTransport(NodeTransport):
    def __init__(self, services: Dict[str, StorageService]):
        self.services = services

    async def request_por(self, node_id: str, challenge: PorChallenge) -> PorResponse:
        svc = self.services[node_id]
        return svc.answer_por_challenge(challenge)


@dataclass(frozen=True)
class VerificationConfig:
    challenge_interval_seconds: float = 30.0


class VerificationOrchestrator:
    def __init__(
        self,
        *,
        node_id: str,
        ledger: InMemoryAegisStorageLedger,
        event_bus: InMemoryContractEventBus,
        transport: NodeTransport,
        store: Optional[StorageService] = None,
        config: Optional[VerificationConfig] = None,
    ):
        self.node_id = node_id
        self.ledger = ledger
        self.bus = event_bus
        self.transport = transport
        self.store = store
        self.config = config or VerificationConfig()
        self._task: Optional[asyncio.Task] = None
        self._pending: Dict[str, asyncio.Task] = {}

    def start(self) -> None:
        self.bus.subscribe(self._on_event)
        if self.store is not None and self._task is None:
            self._task = asyncio.create_task(self._periodic_self_checks())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        for t in list(self._pending.values()):
            t.cancel()
        self._pending.clear()

    async def _on_event(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "OperationFinalized":
            return
        if event.get("operation_type") != "INTEGRITY_CHALLENGE":
            return

        op_hash = str(event.get("operation_hash"))
        if op_hash in self._pending:
            return
        self._pending[op_hash] = asyncio.create_task(self._handle_integrity_challenge(event))

    async def _handle_integrity_challenge(self, event: Dict[str, Any]) -> None:
        try:
            op = event["operation"]
            challenge_id = op["challenge_id"]
            fragment_hash = op["fragment_hash"]
            storage_node_id = op["storage_node_id"]
            block_index = int(op["block_index"])
            nonce_b64 = op["nonce_b64"]
            expected_root = op["block_merkle_root_hex"]

            challenge = PorChallenge(
                fragment_hash_hex=fragment_hash,
                block_index=block_index,
                nonce_b64=nonce_b64,
                expected_block_merkle_root_hex=expected_root,
            )

            success = False
            response_hash_hex = ""
            try:
                resp = await self.transport.request_por(storage_node_id, challenge)
                verifier = self.store or None
                if verifier is None:
                    verifier = StorageService(base_dir=os.path.join(os.getcwd(), ".tmp_verifier"))
                success = verifier.verify_por_response_locally(challenge, resp)
                response_hash_hex = resp.response_hash_hex
            except Exception:
                success = False
                response_hash_hex = ""

            result_op = {
                "challenge_id": challenge_id,
                "success": bool(success),
                "response_hash": response_hash_hex or "0" * 64,
            }
            await self.ledger.request("INTEGRITY_RESULT", result_op)
        finally:
            op_hash = str(event.get("operation_hash"))
            self._pending.pop(op_hash, None)

    async def _periodic_self_checks(self) -> None:
        while True:
            await asyncio.sleep(self.config.challenge_interval_seconds)
            if self.store is None:
                continue
            try:
                metas = list(self.store.meta_dir.glob("*.json"))
                if not metas:
                    continue
                meta_path = metas[int.from_bytes(os.urandom(2), "big") % len(metas)]
                meta = self.store.get_metadata(meta_path.stem)
                frag_hash = meta.fragment_hash_hex
                chall = self.store.create_por_challenge(frag_hash)
                op = {
                    "challenge_id": _sha256_hex(os.urandom(32)),
                    "file_id": meta.file_id,
                    "fragment_hash": frag_hash,
                    "auditor_id": self.node_id,
                    "storage_node_id": self.node_id,
                    "nonce_b64": chall.nonce_b64,
                    "block_index": chall.block_index,
                    "block_merkle_root_hex": chall.expected_block_merkle_root_hex,
                }
                await self.ledger.request("INTEGRITY_CHALLENGE", op)
            except Exception:
                continue


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()

