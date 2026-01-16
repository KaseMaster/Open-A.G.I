import argparse
import json
import sys
from typing import Any, Dict

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def _verify_proof(leaf_hash_hex: str, index: int, siblings: Any, root_hex: str) -> bool:
    from aegis_storage_audit import merkle_verify_hash

    leaf = bytes.fromhex(leaf_hash_hex)
    root = bytes.fromhex(root_hex)
    proof = [(bytes.fromhex(str(s["hash"])), str(s["position"])) for s in siblings]
    return merkle_verify_hash(leaf, proof, root)


def _leaf_hash_of_event(ev: Dict[str, Any]) -> str:
    from aegis_storage_audit import _canonical_json_bytes, _sha256

    return _sha256(_canonical_json_bytes(ev)).hex()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True, help="Ruta a JSON del evento (canonical_dict)")
    p.add_argument("--proof", required=True, help="Ruta a JSON de la prueba (siblings+index+root)")
    p.add_argument("--pubkey-hex", required=True, help="Ed25519 public key hex")
    args = p.parse_args()

    ev = json.loads(open(args.event, "r", encoding="utf-8").read())
    proof = json.loads(open(args.proof, "r", encoding="utf-8").read())

    leaf = _leaf_hash_of_event(ev)
    if leaf != str(proof.get("leaf_hash")):
        print("invalid: leaf_hash mismatch")
        sys.exit(2)

    ok_merkle = _verify_proof(leaf, int(proof["index"]), proof["siblings"], str(proof["root"]))
    if not ok_merkle:
        print("invalid: merkle proof failed")
        sys.exit(3)

    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(args.pubkey_hex))
    try:
        import base64
        from aegis_storage_audit import _canonical_json_bytes

        msg = _canonical_json_bytes(
            {
                "id": ev["id"],
                "timestamp": ev["timestamp"],
                "actor_id": ev["actor_id"],
                "event_type": ev["event_type"],
                "status": ev["status"],
                "payload": ev["payload"],
            }
        )
        sig = base64.b64decode(ev["signature_b64"].encode("ascii"))
        pub.verify(sig, msg)
    except Exception:
        print("invalid: signature failed")
        sys.exit(4)

    print("valid")


if __name__ == "__main__":
    main()

