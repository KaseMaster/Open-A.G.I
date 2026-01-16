import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aegis_core.consensus_protocol import PBFTConsensus


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _op_hash(operation: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(operation)).hexdigest()


class InMemoryContractEventBus:
    def __init__(self):
        self._subscribers: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []

    def subscribe(self, fn: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._subscribers.append(fn)

    async def emit(self, event: Dict[str, Any]) -> None:
        for fn in list(self._subscribers):
            await fn(event)


@dataclass
class LedgerOperation:
    operation_type: str
    operation: Dict[str, Any]
    operation_hash: str


class InMemoryAegisStorageLedger:
    def __init__(self, event_bus: InMemoryContractEventBus):
        self.event_bus = event_bus
        self.finalized_ops: Dict[str, bool] = {}
        self.files: Dict[str, Dict[str, Any]] = {}
        self.fragment_locations: Dict[str, List[str]] = {}
        self.access_grants: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.challenges: Dict[str, Dict[str, Any]] = {}

    async def request(self, operation_type: str, operation: Dict[str, Any]) -> LedgerOperation:
        op_hash = _op_hash({"type": operation_type, "op": operation})
        evt = {
            "event": "ConsensusRequired",
            "operation_hash": op_hash,
            "operation_type": operation_type,
            "operation": operation,
        }
        await self.event_bus.emit(evt)
        return LedgerOperation(operation_type=operation_type, operation=operation, operation_hash=op_hash)

    def finalize(self, operation_type: str, operation: Dict[str, Any], operation_hash: str) -> None:
        expected = _op_hash({"type": operation_type, "op": operation})
        if expected != operation_hash:
            raise ValueError("op hash mismatch")
        if self.finalized_ops.get(operation_hash):
            raise ValueError("op already finalized")

        if operation_type == "FILE_UPLOAD":
            file_id = operation["file_id"]
            if file_id in self.files:
                raise ValueError("file exists")
            self.files[file_id] = dict(operation)

        elif operation_type == "FRAGMENT_LOCATION":
            frag = operation["fragment_hash"]
            node = operation["storage_node_id"]
            self.fragment_locations.setdefault(frag, [])
            if node not in self.fragment_locations[frag]:
                self.fragment_locations[frag].append(node)

        elif operation_type == "ACCESS_GRANT":
            file_id = operation["file_id"]
            grantee = operation["grantee_id"]
            self.access_grants.setdefault(file_id, {})
            if grantee in self.access_grants[file_id]:
                raise ValueError("grant immutable")
            self.access_grants[file_id][grantee] = dict(operation)

        elif operation_type == "INTEGRITY_CHALLENGE":
            challenge_id = operation["challenge_id"]
            if challenge_id in self.challenges:
                raise ValueError("challenge exists")
            self.challenges[challenge_id] = dict(operation)
            self.challenges[challenge_id]["status"] = "OPEN"

        elif operation_type == "INTEGRITY_RESULT":
            challenge_id = operation["challenge_id"]
            if challenge_id not in self.challenges:
                raise ValueError("missing challenge")
            if self.challenges[challenge_id].get("status") != "OPEN":
                raise ValueError("already resolved")
            self.challenges[challenge_id]["status"] = "RESOLVED"
            self.challenges[challenge_id]["success"] = bool(operation["success"])
            self.challenges[challenge_id]["response_hash"] = operation["response_hash"]
            self.challenges[challenge_id]["resolved_at"] = time.time()

        else:
            raise ValueError("unknown operation_type")

        self.finalized_ops[operation_hash] = True


class InMemoryPBFTNetwork:
    def __init__(self):
        self.nodes: Dict[str, PBFTConsensus] = {}
        self.crypto_engine = None

    def register(self, node_id: str, pbft: PBFTConsensus) -> None:
        self.nodes[node_id] = pbft

    async def broadcast_message(self, _net_type: Any, payload: Dict[str, Any]) -> None:
        sender_id = payload.get("sender_id")
        for node_id, pbft in list(self.nodes.items()):
            await pbft._on_consensus_network_message(sender_id or node_id, {"payload": payload})

    def register_handler(self, _net_type: Any, _handler: Any) -> None:
        return


class StorageConsensusAdapter:
    def __init__(
        self,
        *,
        validators: Dict[str, Any],
        event_bus: InMemoryContractEventBus,
        ledger: InMemoryAegisStorageLedger,
        validator_policy: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.event_bus = event_bus
        self.ledger = ledger
        self.validator_policy = validator_policy

        self.network = InMemoryPBFTNetwork()
        self.pbft_nodes: Dict[str, PBFTConsensus] = {}
        for node_id, priv in validators.items():
            pbft = PBFTConsensus(node_id=node_id, private_key=priv, network_manager=self.network)
            self.pbft_nodes[node_id] = pbft
            self.network.register(node_id, pbft)

        for pbft in self.pbft_nodes.values():
            for other in self.pbft_nodes.values():
                pbft.add_node(other.node_id, other.public_key)

        self.event_bus.subscribe(self._on_contract_event)

    def _leader_id(self) -> str:
        node0 = next(iter(self.pbft_nodes.values()))
        nodes_by_reputation = sorted(
            node0.node_reputations.items(),
            key=lambda x: (x[1].computation_score + x[1].reliability_score, x[0]),
        )
        leader_index = node0.view_number % len(nodes_by_reputation)
        return nodes_by_reputation[leader_index][0]

    async def _on_contract_event(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "ConsensusRequired":
            return

        operation_hash = event["operation_hash"]
        operation_type = event["operation_type"]
        operation = event["operation"]

        change_data = {
            "type": "aegis_storage_ledger_op",
            "operation_hash": operation_hash,
            "operation_type": operation_type,
            "operation": operation,
        }

        if self.validator_policy is not None:
            leader = self._leader_id()
            change_data = self.validator_policy(leader, change_data)

        await self.run_pbft_and_finalize(change_data)

    async def run_pbft_and_finalize(self, change_data: Dict[str, Any]) -> None:
        leader_id = self._leader_id()
        leader = self.pbft_nodes[leader_id]

        ok = await leader.propose_change(change_data)
        if not ok:
            raise RuntimeError("leader no pudo proponer")

        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if leader.state.name == "IDLE":
                break
            await asyncio.sleep(0.02)

        op_hash = change_data["operation_hash"]
        op_type = change_data["operation_type"]
        op = change_data["operation"]
        self.ledger.finalize(op_type, op, op_hash)

        await self.event_bus.emit(
            {
                "event": "OperationFinalized",
                "operation_hash": op_hash,
                "operation_type": op_type,
                "operation": op,
            }
        )

