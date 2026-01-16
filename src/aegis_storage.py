from typing import Any, Dict, List, Optional

from aegis_core.crypto_framework import CryptoEngine, initialize_crypto

from aegis_storage_audit import MerkleAuditLog, create_audit_event


class AegisStorageRuntime:
    def __init__(self):
        self.storage_node = None
        self.storage_service = None
        self.network_client = None
        self.tor_gateway = None
        self.dashboard_runner = None


class AegisStorageDApp:
    def __init__(
        self,
        *,
        node_id: str,
        security_level: str = "HIGH",
        audit_persist_path: Optional[str] = None,
    ):
        self.crypto: CryptoEngine = initialize_crypto(
            {"security_level": security_level, "node_id": node_id}
        )
        self.audit = MerkleAuditLog(persist_path=audit_persist_path)
        self._event_id_to_index: Dict[str, int] = {}
        self.runtime = AegisStorageRuntime()

        self.ledger = None
        self.ledger_bus = None

    def record_event(self, *, event_type: str, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        ev = create_audit_event(crypto=self.crypto, event_type=event_type, status=status, payload=payload)
        idx = self.audit.append(ev)
        self._event_id_to_index[ev.id] = idx
        return {"id": ev.id, "index": idx, "leaf_hash": ev.leaf_hash().hex(), "root": self.audit.current_root()}

    def bind_storage_ledger(self, *, ledger: Any, event_bus: Any) -> None:
        self.ledger = ledger
        self.ledger_bus = event_bus

        async def _on_evt(evt: Dict[str, Any]) -> None:
            if evt.get("event") != "OperationFinalized":
                return
            op_type = str(evt.get("operation_type"))
            op = evt.get("operation") or {}
            self.record_event(event_type=f"ledger:{op_type}", status="ok", payload={"op": op})

        event_bus.subscribe(_on_evt)

    async def start_storage_node(self, *, storage_dir: str, host: str = "127.0.0.1", port: int = 8088) -> None:
        from aegis_storage_node.api_server import ServerConfig, StorageNodeAPIServer
        from aegis_storage_node.storage_service import StorageService

        svc = StorageService(storage_dir)
        server = StorageNodeAPIServer(
            node_id=self.crypto.identity.node_id if self.crypto.identity else "storage_node",
            crypto_engine=self.crypto,
            storage=svc,
            config=ServerConfig(host=host, port=port),
        )
        await server.start()
        self.runtime.storage_node = server
        self.runtime.storage_service = svc
        self.record_event(event_type="node:start", status="ok", payload={"host": host, "port": port})

    async def stop_storage_node(self) -> None:
        server = self.runtime.storage_node
        if server is not None:
            await server.stop()
        self.runtime.storage_node = None
        self.runtime.storage_service = None

    async def init_network_client(self, *, tor_control_port: int = 9051, tor_socks_port: int = 9050) -> None:
        from aegis_core.tor_integration import TorGateway
        from aegis_storage_client.network_client import AegisStorageNetworkClient

        tor = TorGateway(control_port=int(tor_control_port), socks_port=int(tor_socks_port))
        ok = await tor.initialize()
        if not ok:
            raise RuntimeError("tor init failed")
        net = AegisStorageNetworkClient(crypto_engine=self.crypto, tor_gateway=tor)
        self.runtime.tor_gateway = tor
        self.runtime.network_client = net
        self.record_event(event_type="client:init", status="ok", payload={"tor": True})

    def get_audit_status(self) -> Dict[str, Any]:
        return {
            "events": self.audit.count,
            "root": self.audit.current_root(),
        }

    def list_audit_events(self, *, query: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        return self.audit.list_events(query=query, limit=limit)

    def get_audit_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        idx = self._event_id_to_index.get(event_id)
        if idx is None:
            return None
        ev = self.audit.get_event(idx)
        d = ev.canonical_dict()
        d["index"] = idx
        d["leaf_hash"] = ev.leaf_hash().hex()
        return d

    def get_audit_proof(self, event_id: str) -> Optional[Dict[str, Any]]:
        idx = self._event_id_to_index.get(event_id)
        if idx is None:
            return None
        return self.audit.proof(idx)

    def verify_audit_proof(self, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            computed = self.audit.compute_root_from_request(
                leaf_hash_hex=str(req["leaf_hash"]),
                siblings=req.get("siblings", []),
            )
            ok = self.audit.verify_proof(
                leaf_hash_hex=str(req["leaf_hash"]),
                index=int(req["index"]),
                siblings=req.get("siblings", []),
                root_hex=str(req["root"]),
            )
            return {
                "valid": bool(ok),
                "computedRoot": computed,
                "reason": None if ok else "root_mismatch",
            }
        except Exception as e:
            return {"valid": False, "computedRoot": "", "reason": str(e)}

