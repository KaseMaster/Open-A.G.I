import base64
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
DAPP_SRC = os.path.join(PROJECT_ROOT, "dapps", "aegis-storage", "src")
for p in [PROJECT_ROOT, SRC_ROOT, DAPP_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _to_handshake_public_identity(public_identity_bytes: dict) -> dict:
    return {
        "node_id": base64.b64encode(public_identity_bytes["node_id"]).decode("ascii"),
        "signing_key": base64.b64encode(public_identity_bytes["signing_key"]).decode("ascii"),
        "encryption_key": base64.b64encode(public_identity_bytes["encryption_key"]).decode("ascii"),
        "created_at": base64.b64encode(public_identity_bytes["created_at"]).decode("ascii"),
    }


def test_storage_tx_signature_compatible_with_core_peer_identity():
    from identity_manager import StorageIdentityManager, StorageRole

    alice = StorageIdentityManager(node_id="alice")
    bob = StorageIdentityManager(node_id="bob")

    alice.add_peer_identity(bob.identity.export_public_identity())
    bob.add_peer_identity(alice.identity.export_public_identity())

    tx = alice.request_role(StorageRole.DATA_OWNER, {"proof": "k"})
    assert bob.verify_storage_tx(tx) is True


def test_handshake_public_identity_roundtrip_adds_peer_and_secure_channel():
    from aegis_core.crypto_framework import create_crypto_engine, SecurityLevel

    alice = create_crypto_engine(SecurityLevel.HIGH)
    bob = create_crypto_engine(SecurityLevel.HIGH)
    alice.generate_node_identity("alice")
    bob.generate_node_identity("bob")

    bob_pub = _to_handshake_public_identity(bob.identity.export_public_identity())
    peer_public = {
        "node_id": base64.b64decode(bob_pub["node_id"]),
        "signing_key": base64.b64decode(bob_pub["signing_key"]),
        "encryption_key": base64.b64decode(bob_pub["encryption_key"]),
        "created_at": base64.b64decode(bob_pub["created_at"]),
    }

    assert alice.add_peer_identity(peer_public) is True
    assert alice.establish_secure_channel("bob") is True
    assert "bob" in alice.ratchet_states


def test_multisig_bundle_applies_attribute_change_in_state():
    from identity_manager import (
        StorageIdentityManager,
        build_attribute_change_operation,
        collect_multisig_approvals,
    )
    from storage_identity_state import StorageIdentityState

    auditor_1 = StorageIdentityManager(node_id="aud1")
    auditor_2 = StorageIdentityManager(node_id="aud2")
    auditor_3 = StorageIdentityManager(node_id="aud3")

    policy_tx = auditor_1.create_multisig_policy(
        "attribute_change",
        ["aud1", "aud2", "aud3"],
        threshold_m=2,
    )

    state = StorageIdentityState()
    assert state.apply_change(policy_tx.payload) is True

    op = build_attribute_change_operation(
        target_node_id="target",
        attribute="can_store_medical_data",
        value=True,
        policy_id=policy_tx.payload["policy_id"],
    )

    a1 = auditor_1.approve_multisig_op(op_id=op["op_id"], policy_id=op["policy_id"])
    a2 = auditor_2.approve_multisig_op(op_id=op["op_id"], policy_id=op["policy_id"])

    bundle = collect_multisig_approvals(op_payload=op, approvals=[a1, a2])
    assert state.apply_change(bundle) is True
    assert state.roles["target"].attributes["can_store_medical_data"] is True

