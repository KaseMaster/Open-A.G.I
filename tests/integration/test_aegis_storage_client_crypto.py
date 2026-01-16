import os
import random
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_encrypt_fragment_reconstruct_roundtrip_any_k():
    from aegis_storage_client.encryption_engine import (
        EncryptedFile,
        decrypt_file_chacha,
        decrypt_and_reconstruct,
        encrypt_file_chacha,
        encrypt_fragments,
    )

    data = os.urandom(256 * 1024)
    enc = encrypt_file_chacha(data)
    frags, original_len, _root = encrypt_fragments(
        encrypted_file_ciphertext=enc.ciphertext,
        file_key=enc.file_key,
        k=4,
        n=6,
    )

    pick = sorted(random.sample(range(6), 4))
    subset = [frags[i] for i in pick]
    rec_ct = decrypt_and_reconstruct(
        fragments=subset,
        file_key=enc.file_key,
        original_len=original_len,
        k=4,
        n=6,
    )
    dec = decrypt_file_chacha(
        EncryptedFile(file_key=enc.file_key, chacha_nonce=enc.chacha_nonce, ciphertext=rec_ct)
    )
    assert dec == data


def test_merkle_proof_verifies_leaf():
    from aegis_storage_client.encryption_engine import merkle_proof, merkle_root, merkle_verify

    leaves = [os.urandom(128) for _ in range(9)]
    root = merkle_root(leaves)
    for i in [0, 3, 8]:
        proof = merkle_proof(leaves, i)
        assert merkle_verify(leaves[i], proof, root) is True


def test_secure_envelope_extracts_secure_message():
    import base64
    import json

    from aegis_core.crypto_framework import create_crypto_engine, SecurityLevel
    from aegis_storage_client.network_client import decode_secure_envelope, extract_secure_message

    alice = create_crypto_engine(SecurityLevel.HIGH)
    bob = create_crypto_engine(SecurityLevel.HIGH)
    alice.generate_node_identity("alice")
    bob.generate_node_identity("bob")
    alice.add_peer_identity(bob.identity.export_public_identity())
    bob.add_peer_identity(alice.identity.export_public_identity())
    alice.establish_secure_channel("bob")
    bob.establish_secure_channel("alice")

    plaintext = json.dumps({"k": "v"}, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sm = alice.encrypt_message(plaintext, "bob")
    assert sm is not None
    envelope = {
        "type": "secure",
        "sender_id": sm.sender_id,
        "recipient_id": sm.recipient_id,
        "secure_message_b64": base64.b64encode(sm.serialize()).decode("ascii"),
    }
    raw = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
    parsed = decode_secure_envelope(raw)
    recovered = extract_secure_message(parsed)
    out = bob.decrypt_message(recovered)
    assert out == plaintext

