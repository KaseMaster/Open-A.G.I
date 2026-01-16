import asyncio
import os
from typing import Any, Dict

from aegis_core.crypto_framework import initialize_crypto

from aegis_storage_node.api_server import ServerConfig, StorageNodeAPIServer
from aegis_storage_node.storage_service import StorageService


async def _run() -> None:
    node_id = os.environ.get("AEGIS_NODE_ID", "storage_node")
    security = os.environ.get("AEGIS_SECURITY_LEVEL", "HIGH")
    storage_dir = os.environ.get("AEGIS_STORAGE_DIR", "/data")
    host = os.environ.get("AEGIS_STORAGE_HOST", "0.0.0.0")
    port = int(os.environ.get("AEGIS_STORAGE_PORT", "8088"))

    crypto = initialize_crypto({"security_level": security, "node_id": node_id})
    storage = StorageService(storage_dir)
    server = StorageNodeAPIServer(
        node_id=node_id,
        crypto_engine=crypto,
        storage=storage,
        config=ServerConfig(host=host, port=port),
    )
    await server.start()

    while True:
        await asyncio.sleep(3600)


def main() -> None:
    asyncio.run(_run())

