import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from aegis_core.crypto_framework import initialize_crypto
from aegis_core.tor_integration import TorGateway

from aegis_storage_client.encryption_engine import (
    encrypt_file_chacha,
    encrypt_fragments,
)
from aegis_storage_client.network_client import AegisStorageNetworkClient, StorageNodeEndpoint


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _load_nodes(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@click.group()
def main():
    pass


@main.command("upload")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--security", type=click.Choice(["STANDARD", "HIGH", "PARANOID"], case_sensitive=False), default="PARANOID")
@click.option("--k", type=int, default=4)
@click.option("--n", type=int, default=6)
@click.option("--nodes", "nodes_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None)
def upload_cmd(file_path: str, security: str, k: int, n: int, nodes_path: Optional[str], out_path: Optional[str]):
    p = Path(file_path)
    data = p.read_bytes()

    enc = encrypt_file_chacha(data)
    frags, original_len, root = encrypt_fragments(
        encrypted_file_ciphertext=enc.ciphertext,
        file_key=enc.file_key,
        k=k,
        n=n,
    )

    frag_blobs = [f.blob() for f in frags]
    frag_hashes = {i: _sha256_hex(b) for i, b in enumerate(frag_blobs)}
    file_id = _sha256_hex(enc.ciphertext)
    merkle_root_hex = root.hex()

    manifest: Dict[str, Any] = {
        "file_id": file_id,
        "source_name": p.name,
        "created_at": time.time(),
        "security": security.upper(),
        "k": k,
        "n": n,
        "chacha_nonce_hex": enc.chacha_nonce.hex(),
        "file_key_hex": enc.file_key.hex(),
        "ciphertext_len": len(enc.ciphertext),
        "rs_original_len": original_len,
        "merkle_root_hex": merkle_root_hex,
        "fragment_hashes": {str(i): h for i, h in frag_hashes.items()},
    }

    nodes_cfg = _load_nodes(nodes_path)
    targets = nodes_cfg.get("storage_nodes") or []
    if targets:
        crypto = initialize_crypto({"security_level": security.upper(), "node_id": nodes_cfg.get("node_id", "client")})
        tor = TorGateway(
            control_port=int(nodes_cfg.get("tor_control_port", 9051)),
            socks_port=int(nodes_cfg.get("tor_socks_port", 9050)),
        )
        net = AegisStorageNetworkClient(crypto_engine=crypto, tor_gateway=tor, security_level=security.upper())
        ok = asyncio.run(net.initialize())
        if not ok:
            raise SystemExit(2)
        for node in targets:
            peer_id = str(node["node_id"])
            if "public_identity" in node:
                crypto.add_peer_identity({
                    "node_id": node["public_identity"]["node_id"].encode(),
                    "signing_key": bytes.fromhex(node["public_identity"]["signing_key_hex"]),
                    "encryption_key": bytes.fromhex(node["public_identity"]["encryption_key_hex"]),
                    "created_at": node["public_identity"]["created_at"].encode(),
                })
            target = StorageNodeEndpoint(node_id=peer_id, onion=str(node["onion"]), port=int(node.get("port", 8088)))
            upload_map = {i: frag_blobs[i] for i in range(len(frag_blobs))}
            ok = asyncio.run(
                net.upload_fragments(
                    target=target,
                    file_id=file_id,
                    fragments=upload_map,
                    merkle_root_hex=merkle_root_hex,
                    fragment_hashes_hex=frag_hashes,
                )
            )
            if not ok:
                raise SystemExit(3)

    out = out_path or (str(p) + ".aegis-storage.json")
    Path(out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(out)


@main.command("grant-access")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--to", "to_identity", required=True)
@click.option("--expiry", default="7d")
def grant_access_cmd(manifest_path: str, to_identity: str, expiry: str):
    p = Path(manifest_path)
    m = json.loads(p.read_text(encoding="utf-8"))
    grant = {
        "file_id": m["file_id"],
        "to": to_identity,
        "expiry": expiry,
    }
    out = p.with_suffix(p.suffix + ".grant.json")
    out.write_text(json.dumps(grant, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(str(out))

