import asyncio
import os
import sys
import time

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.mark.asyncio
async def test_full_network_flow(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from pbft_integration import InMemoryAegisStorageLedger, InMemoryContractEventBus, StorageConsensusAdapter
    from aegis_storage_client.encryption_engine import (
        EncryptedFile,
        decrypt_file_chacha,
        decrypt_and_reconstruct,
        encrypt_file_chacha,
        encrypt_fragments,
        parse_fragment_blob,
    )
    from aegis_storage_node.storage_service import StorageService
    from aegis_storage_node.verification_orchestrator import InMemoryNodeTransport, VerificationOrchestrator
    from aegis_storage_audit import MerkleAuditLog, create_audit_event
    from aegis_core.crypto_framework import create_crypto_engine, SecurityLevel

    t0 = time.time()
    bus = InMemoryContractEventBus()
    ledger = InMemoryAegisStorageLedger(bus)

    validators = {f"v{i}": ed25519.Ed25519PrivateKey.generate() for i in range(1, 5)}
    _adapter = StorageConsensusAdapter(validators=validators, event_bus=bus, ledger=ledger)

    class ByzantineStore(StorageService):
        def answer_por_challenge(self, challenge):
            resp = super().answer_por_challenge(challenge)
            d = dict(resp.__dict__)
            d["block_b64"] = d["block_b64"][:-4] + "AAAA"
            return type(resp)(**d)

    nodes = {
        "n1": StorageService(str(tmp_path / "n1")),
        "n2": StorageService(str(tmp_path / "n2")),
        "n3": StorageService(str(tmp_path / "n3")),
        "n4": ByzantineStore(str(tmp_path / "n4")),
    }
    transport = InMemoryNodeTransport(nodes)

    auditor_store = StorageService(str(tmp_path / "aud"))
    orchestrator = VerificationOrchestrator(
        node_id="auditor",
        ledger=ledger,
        event_bus=bus,
        transport=transport,
        store=auditor_store,
    )
    orchestrator.start()

    owner1 = create_crypto_engine(SecurityLevel.HIGH)
    owner2 = create_crypto_engine(SecurityLevel.HIGH)
    owner1.generate_node_identity("owner1")
    owner2.generate_node_identity("owner2")

    audit = MerkleAuditLog(persist_path=str(tmp_path / "audit.log"))

    data = b"secret_document" * 20000
    enc = encrypt_file_chacha(data)
    frags, original_len, _root = encrypt_fragments(
        encrypted_file_ciphertext=enc.ciphertext,
        file_key=enc.file_key,
        k=4,
        n=6,
    )
    frag_blobs = [f.blob() for f in frags]
    frag_hashes = [__import__("hashlib").sha256(b).hexdigest() for b in frag_blobs]
    file_id = __import__("hashlib").sha256(enc.ciphertext).hexdigest()

    ev = create_audit_event(
        crypto=owner1,
        event_type="upload",
        status="ok",
        payload={"file_id": file_id, "k": 4, "n": 6},
    )
    audit.append(ev)

    await ledger.request(
        "FILE_UPLOAD",
        {
            "file_id": file_id,
            "owner_id": "owner1",
            "file_root_hash": file_id,
            "file_size": len(enc.ciphertext),
            "metadata_hash": "0" * 64,
            "fragment_hashes": frag_hashes,
        },
    )
    await asyncio.sleep(0.35)

    placement = ["n1", "n2", "n3", "n4", "n1", "n2"]
    for i, node_id in enumerate(placement):
        nodes[node_id].put_fragment(
            owner_id="owner1",
            file_id=file_id,
            fragment_index=i,
            fragment_blob=frag_blobs[i],
            file_merkle_root_hex="0" * 64,
            fragment_hash_hex=frag_hashes[i],
        )
        await ledger.request(
            "FRAGMENT_LOCATION",
            {"file_id": file_id, "fragment_hash": frag_hashes[i], "storage_node_id": node_id},
        )
    await asyncio.sleep(0.35)

    await ledger.request(
        "ACCESS_GRANT",
        {
            "file_id": file_id,
            "owner_id": "owner1",
            "grantee_id": "owner2",
            "permissions_mask": 0b0001,
            "expires_at": int(time.time()) + 3600,
        },
    )
    await asyncio.sleep(0.35)

    honest = {k: v for k, v in nodes.items() if k != "n4"}
    chosen = [0, 1, 2, 4]
    subset_frags = []
    for idx in chosen:
        node_id = placement[idx]
        if node_id == "n4":
            node_id = "n1"
        blob = honest[node_id].get_fragment(frag_hashes[idx])
        subset_frags.append(parse_fragment_blob(blob))

    rec_ct = decrypt_and_reconstruct(
        fragments=subset_frags,
        file_key=enc.file_key,
        original_len=original_len,
        k=4,
        n=6,
    )
    dec = decrypt_file_chacha(EncryptedFile(file_key=enc.file_key, chacha_nonce=enc.chacha_nonce, ciphertext=rec_ct))
    assert dec == data

    byz_fail = 0
    for frag_hash in frag_hashes:
        if not nodes["n4"].has_fragment(frag_hash):
            continue
        meta = nodes["n4"].get_metadata(frag_hash)
        chall = nodes["n4"].create_por_challenge(frag_hash, block_index=0)
        op = {
            "challenge_id": "ch_" + frag_hash[:12],
            "file_id": meta.file_id,
            "fragment_hash": frag_hash,
            "auditor_id": "auditor",
            "storage_node_id": "n4",
            "nonce_b64": chall.nonce_b64,
            "block_index": chall.block_index,
            "block_merkle_root_hex": meta.block_merkle_root_hex,
        }
        await ledger.request("INTEGRITY_CHALLENGE", op)
        for _ in range(40):
            await asyncio.sleep(0.05)
            ch = ledger.challenges.get(op["challenge_id"], {})
            if ch.get("status") == "RESOLVED":
                break
        if ledger.challenges[op["challenge_id"]].get("success") is False:
            byz_fail += 1
    assert byz_fail >= 1

    elapsed = time.time() - t0
    assert elapsed < 20

