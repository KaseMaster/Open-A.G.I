import asyncio
import os
import sys
import time

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _k32(label: str) -> str:
    return (label + "_" * 64)[:64]


@pytest.mark.asyncio
async def test_pbft_finalizes_file_upload_and_sharding_locations():
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from pbft_integration import (
        InMemoryAegisStorageLedger,
        InMemoryContractEventBus,
        StorageConsensusAdapter,
    )

    bus = InMemoryContractEventBus()
    ledger = InMemoryAegisStorageLedger(bus)

    validators = {
        "v1": ed25519.Ed25519PrivateKey.generate(),
        "v2": ed25519.Ed25519PrivateKey.generate(),
        "v3": ed25519.Ed25519PrivateKey.generate(),
        "v4": ed25519.Ed25519PrivateKey.generate(),
    }
    _ = StorageConsensusAdapter(validators=validators, event_bus=bus, ledger=ledger)

    op = {
        "file_id": _k32("fileA"),
        "owner_id": _k32("owner"),
        "file_root_hash": _k32("root"),
        "file_size": 123,
        "metadata_hash": _k32("meta"),
        "fragment_hashes": [_k32("f1"), _k32("f2"), _k32("f3")],
    }
    await ledger.request("FILE_UPLOAD", op)
    await asyncio.sleep(0.35)

    assert op["file_id"] in ledger.files
    assert ledger.files[op["file_id"]]["fragment_hashes"] == op["fragment_hashes"]

    loc = {
        "file_id": op["file_id"],
        "fragment_hash": op["fragment_hashes"][0],
        "storage_node_id": _k32("nodeX"),
    }
    await ledger.request("FRAGMENT_LOCATION", loc)
    await asyncio.sleep(0.35)
    assert _k32("nodeX") in ledger.fragment_locations[op["fragment_hashes"][0]]


@pytest.mark.asyncio
async def test_byzantine_storage_node_false_loss_cannot_overwrite_integrity_result():
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from pbft_integration import (
        InMemoryAegisStorageLedger,
        InMemoryContractEventBus,
        StorageConsensusAdapter,
        _op_hash,
    )

    bus = InMemoryContractEventBus()
    ledger = InMemoryAegisStorageLedger(bus)

    validators = {
        "v1": ed25519.Ed25519PrivateKey.generate(),
        "v2": ed25519.Ed25519PrivateKey.generate(),
        "v3": ed25519.Ed25519PrivateKey.generate(),
        "v4": ed25519.Ed25519PrivateKey.generate(),
    }
    _ = StorageConsensusAdapter(validators=validators, event_bus=bus, ledger=ledger)

    challenge = {
        "challenge_id": _k32("ch1"),
        "file_id": _k32("fileA"),
        "fragment_hash": _k32("frag"),
        "auditor_id": _k32("aud"),
        "storage_node_id": _k32("node"),
        "nonce": _k32("nonce"),
    }
    await ledger.request("INTEGRITY_CHALLENGE", challenge)
    await asyncio.sleep(0.35)
    assert ledger.challenges[challenge["challenge_id"]]["status"] == "OPEN"

    result_ok = {
        "challenge_id": challenge["challenge_id"],
        "success": True,
        "response_hash": _k32("resp_ok"),
    }
    await ledger.request("INTEGRITY_RESULT", result_ok)
    await asyncio.sleep(0.35)

    assert ledger.challenges[challenge["challenge_id"]]["status"] == "RESOLVED"
    assert ledger.challenges[challenge["challenge_id"]]["success"] is True

    malicious_result = {
        "challenge_id": challenge["challenge_id"],
        "success": False,
        "response_hash": _k32("resp_lie"),
    }
    op_hash = _op_hash({"type": "INTEGRITY_RESULT", "op": malicious_result})
    with pytest.raises(ValueError):
        ledger.finalize("INTEGRITY_RESULT", malicious_result, op_hash)


@pytest.mark.asyncio
async def test_byzantine_owner_cannot_modify_access_grant_after_consensus():
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from pbft_integration import (
        InMemoryAegisStorageLedger,
        InMemoryContractEventBus,
        StorageConsensusAdapter,
    )

    bus = InMemoryContractEventBus()
    ledger = InMemoryAegisStorageLedger(bus)

    validators = {
        "v1": ed25519.Ed25519PrivateKey.generate(),
        "v2": ed25519.Ed25519PrivateKey.generate(),
        "v3": ed25519.Ed25519PrivateKey.generate(),
        "v4": ed25519.Ed25519PrivateKey.generate(),
    }
    _ = StorageConsensusAdapter(validators=validators, event_bus=bus, ledger=ledger)

    file_op = {
        "file_id": _k32("fileB"),
        "owner_id": _k32("owner"),
        "file_root_hash": _k32("root"),
        "file_size": 1,
        "metadata_hash": _k32("meta"),
        "fragment_hashes": [_k32("f1")],
    }
    await ledger.request("FILE_UPLOAD", file_op)
    await asyncio.sleep(0.35)

    grant = {
        "file_id": file_op["file_id"],
        "owner_id": file_op["owner_id"],
        "grantee_id": _k32("userX"),
        "permissions_mask": 0b0011,
        "expires_at": int(time.time()) + 3600,
    }
    await ledger.request("ACCESS_GRANT", grant)
    await asyncio.sleep(0.35)

    assert ledger.access_grants[file_op["file_id"]][grant["grantee_id"]]["permissions_mask"] == 0b0011

    modified = dict(grant)
    modified["permissions_mask"] = 0b1111
    with pytest.raises(ValueError):
        await ledger.request("ACCESS_GRANT", modified)

    assert ledger.access_grants[file_op["file_id"]][grant["grantee_id"]]["permissions_mask"] == 0b0011

