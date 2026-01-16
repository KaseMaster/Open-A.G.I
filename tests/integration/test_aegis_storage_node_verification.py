import asyncio
import os
import sys

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.mark.asyncio
async def test_verification_orchestrator_records_integrity_result_success(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from pbft_integration import InMemoryAegisStorageLedger, InMemoryContractEventBus, StorageConsensusAdapter
    from aegis_storage_node.storage_service import StorageService
    from aegis_storage_node.verification_orchestrator import InMemoryNodeTransport, VerificationOrchestrator

    bus = InMemoryContractEventBus()
    ledger = InMemoryAegisStorageLedger(bus)

    validators = {f"v{i}": ed25519.Ed25519PrivateKey.generate() for i in range(1, 5)}
    _adapter = StorageConsensusAdapter(validators=validators, event_bus=bus, ledger=ledger)

    node_store = StorageService(str(tmp_path / "node_store"))
    frag_blob = os.urandom(32 * 1024)
    frag_hash = __import__("hashlib").sha256(frag_blob).hexdigest()
    meta = node_store.put_fragment(
        owner_id="owner",
        file_id="file",
        fragment_index=0,
        fragment_blob=frag_blob,
        file_merkle_root_hex="0" * 64,
        fragment_hash_hex=frag_hash,
    )

    transport = InMemoryNodeTransport({"nodeA": node_store})
    orchestrator = VerificationOrchestrator(
        node_id="auditor",
        ledger=ledger,
        event_bus=bus,
        transport=transport,
        store=StorageService(str(tmp_path / "aud_store")),
    )
    orchestrator.start()

    chall = node_store.create_por_challenge(frag_hash, block_index=0)
    op = {
        "challenge_id": "ch1",
        "file_id": "file",
        "fragment_hash": frag_hash,
        "auditor_id": "auditor",
        "storage_node_id": "nodeA",
        "nonce_b64": chall.nonce_b64,
        "block_index": chall.block_index,
        "block_merkle_root_hex": meta.block_merkle_root_hex,
    }
    await ledger.request("INTEGRITY_CHALLENGE", op)

    for _ in range(30):
        await asyncio.sleep(0.05)
        ch = ledger.challenges.get("ch1")
        if ch and ch.get("status") == "RESOLVED":
            break

    assert ledger.challenges["ch1"]["status"] == "RESOLVED"
    assert ledger.challenges["ch1"]["success"] is True


@pytest.mark.asyncio
async def test_verification_orchestrator_marks_failure_for_byzantine_node(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from pbft_integration import InMemoryAegisStorageLedger, InMemoryContractEventBus, StorageConsensusAdapter
    from aegis_storage_node.storage_service import StorageService
    from aegis_storage_node.verification_orchestrator import InMemoryNodeTransport, VerificationOrchestrator

    class FaultyStore(StorageService):
        def answer_por_challenge(self, challenge):
            resp = super().answer_por_challenge(challenge)
            d = dict(resp.__dict__)
            d["block_b64"] = d["block_b64"][:-4] + "AAAA"
            return type(resp)(**d)

    bus = InMemoryContractEventBus()
    ledger = InMemoryAegisStorageLedger(bus)
    validators = {f"v{i}": ed25519.Ed25519PrivateKey.generate() for i in range(1, 5)}
    _adapter = StorageConsensusAdapter(validators=validators, event_bus=bus, ledger=ledger)

    node_store = FaultyStore(str(tmp_path / "node_store"))
    frag_blob = os.urandom(16 * 1024)
    frag_hash = __import__("hashlib").sha256(frag_blob).hexdigest()
    meta = node_store.put_fragment(
        owner_id="owner",
        file_id="file",
        fragment_index=0,
        fragment_blob=frag_blob,
        file_merkle_root_hex="0" * 64,
        fragment_hash_hex=frag_hash,
    )

    transport = InMemoryNodeTransport({"nodeA": node_store})
    orchestrator = VerificationOrchestrator(
        node_id="auditor",
        ledger=ledger,
        event_bus=bus,
        transport=transport,
        store=StorageService(str(tmp_path / "aud_store")),
    )
    orchestrator.start()

    chall = node_store.create_por_challenge(frag_hash, block_index=0)
    op = {
        "challenge_id": "ch1",
        "file_id": "file",
        "fragment_hash": frag_hash,
        "auditor_id": "auditor",
        "storage_node_id": "nodeA",
        "nonce_b64": chall.nonce_b64,
        "block_index": chall.block_index,
        "block_merkle_root_hex": meta.block_merkle_root_hex,
    }
    await ledger.request("INTEGRITY_CHALLENGE", op)

    for _ in range(30):
        await asyncio.sleep(0.05)
        ch = ledger.challenges.get("ch1")
        if ch and ch.get("status") == "RESOLVED":
            break

    assert ledger.challenges["ch1"]["status"] == "RESOLVED"
    assert ledger.challenges["ch1"]["success"] is False

