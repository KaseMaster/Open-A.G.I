import base64
import os
from pathlib import Path
import argparse

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization


def b32_nopad(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").rstrip("=")


def main():
    parser = argparse.ArgumentParser(description="Generate Tor v3 client authorization files (X25519)")
    parser.add_argument("-n", "--name", dest="client_name", default="archonadmin", help="Client name label (used for file names)")
    args = parser.parse_args()

    client_name = args.client_name
    project_root = Path(__file__).parent
    hs_dir = project_root / "onion_service"
    auth_dir = hs_dir / "authorized_clients"
    auth_dir.mkdir(parents=True, exist_ok=True)

    # Generate X25519 keypair for client authorization
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

    # Read onion address (without trailing newline)
    hostname_path = hs_dir / "hostname"
    onion = hostname_path.read_text(encoding="utf-8").strip()
    onion_no_tld = onion.replace(".onion", "")

    # Server-side authorized client file
    server_auth_path = auth_dir / f"{client_name}.auth"
    server_auth_path.write_text(f"descriptor:x25519:{pub_b32}\n", encoding="utf-8")

    # Client-side private key helper file (for convenience)
    client_auth_priv = project_root / f"{client_name}.auth_private"
    client_auth_priv.write_text(
        f"{onion_no_tld}:descriptor:x25519:{priv_b32}\n", encoding="utf-8"
    )

    print(f"Generated client authorization for '{client_name}'.")
    print(f"Server-side file: {server_auth_path}")
    print("Contents:")
    print(f"  descriptor:x25519:{pub_b32}")
    print()
    print("Client-side file created:")
    print(f"  {client_auth_priv}")
    print("Contents:")
    print(f"  {onion_no_tld}:descriptor:x25519:{priv_b32}")
    print()
    print("Instructions:")
    print("- On the CLIENT, set ClientOnionAuthDir in torrc and place the .auth_private file there.")
    print("  Example torrc line: ClientOnionAuthDir C:/Users/<you>/tor_onion_auth")
    print(f"  Then move {client_name}.auth_private into that directory and restart Tor.")
    print("- On the SERVER, ensure Tor is restarted so it loads authorized_clients.")


if __name__ == "__main__":
    main()