import argparse
import base64
from pathlib import Path
from typing import List

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization


PROJECT_ROOT = Path(__file__).parent
HS_DIR = PROJECT_ROOT / "onion_service"
AUTH_DIR = HS_DIR / "authorized_clients"
CLIENT_TOKENS_DIR = PROJECT_ROOT


def b32_nopad(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").rstrip("=")


def read_onion() -> str:
    onion = (HS_DIR / "hostname").read_text(encoding="utf-8").strip()
    return onion.replace(".onion", "")


def ensure_dirs():
    AUTH_DIR.mkdir(parents=True, exist_ok=True)


def create_client(name: str) -> None:
    ensure_dirs()
    priv = x25519.X25519PrivateKey.generate()
    pub = priv.public_key()

    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    priv_b32 = b32_nopad(priv_raw)
    pub_b32 = b32_nopad(pub_raw)

    onion_no_tld = read_onion()

    server_auth_path = AUTH_DIR / f"{name}.auth"
    server_auth_path.write_text(f"descriptor:x25519:{pub_b32}\n", encoding="utf-8")

    client_priv_path = CLIENT_TOKENS_DIR / f"{name}.auth_private"
    client_priv_path.write_text(
        f"{onion_no_tld}:descriptor:x25519:{priv_b32}\n", encoding="utf-8"
    )

    print(f"[Created] Server: {server_auth_path}")
    print(f"[Created] Client:  {client_priv_path}")


def list_clients() -> List[str]:
    ensure_dirs()
    names = []
    for p in AUTH_DIR.glob("*.auth"):
        names.append(p.stem)
    print("Authorized clients:")
    for n in sorted(names):
        print(f"- {n}")
    return names


def revoke_client(name: str) -> None:
    target = AUTH_DIR / f"{name}.auth"
    if target.exists():
        target.unlink()
        print(f"[Revoked] Removed: {target}")
        print("Note: Restart Tor (or send SIGNAL HUP) to ensure descriptors are regenerated.")
    else:
        print(f"[Skip] Not found: {target}")


def main():
    parser = argparse.ArgumentParser(description="Manage Tor v3 client authorizations")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Create new authorized client")
    p_add.add_argument("name", help="Client name label")

    sub.add_parser("list", help="List authorized clients")

    p_revoke = sub.add_parser("revoke", help="Revoke client authorization")
    p_revoke.add_argument("name", help="Client name label")

    args = parser.parse_args()

    if args.cmd == "add":
        create_client(args.name)
    elif args.cmd == "list":
        list_clients()
    elif args.cmd == "revoke":
        revoke_client(args.name)


if __name__ == "__main__":
    main()