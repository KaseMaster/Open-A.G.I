import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from aiohttp import web

from aegis_core.crypto_framework import CryptoEngine, SecureMessage

from aegis_storage_node.storage_service import PorChallenge, StorageService


def _b64decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _b64encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8088


class StorageNodeAPIServer:
    def __init__(
        self,
        *,
        node_id: str,
        crypto_engine: CryptoEngine,
        storage: StorageService,
        config: Optional[ServerConfig] = None,
    ):
        self.node_id = node_id
        self.crypto = crypto_engine
        self.storage = storage
        self.config = config or ServerConfig()
        self._runner: Optional[web.AppRunner] = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/message", self._handle_message)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_message(self, request: web.Request) -> web.Response:
        body = await request.read()
        try:
            envelope = json.loads(body.decode("utf-8"))
        except Exception:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        sender_id = None
        try:
            secure_b64 = envelope.get("secure_message_b64")
            if not isinstance(secure_b64, str):
                return web.json_response({"ok": False, "error": "missing_secure"}, status=400)
            sm = SecureMessage.deserialize(_b64decode(secure_b64))
            sender_id = sm.sender_id
            plaintext = self.crypto.decrypt_message(sm)
            if plaintext is None:
                return web.json_response({"ok": False, "error": "decrypt_failed"}, status=400)
            msg = json.loads(plaintext.decode("utf-8"))
        except Exception:
            return web.json_response({"ok": False, "error": "bad_secure_payload"}, status=400)

        action = msg.get("action")
        if action == "upload_fragment":
            payload = await self._handle_upload_fragment(msg)
            return self._secure_response(sender_id, payload)
        if action == "integrity_challenge":
            payload = await self._handle_integrity_challenge(msg)
            return self._secure_response(sender_id, payload)
        if action == "get_fragment":
            payload = await self._handle_get_fragment(msg)
            return self._secure_response(sender_id, payload)
        return web.json_response({"ok": False, "error": "unknown_action"}, status=400)

    def _secure_response(self, sender_id: Optional[str], payload: Dict[str, Any]) -> web.Response:
        if not sender_id:
            return web.json_response(payload)
        try:
            plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            sm = self.crypto.encrypt_message(plaintext, sender_id)
            if sm is None:
                return web.json_response(payload)
            envelope = {
                "type": "secure",
                "sender_id": sm.sender_id,
                "recipient_id": sm.recipient_id,
                "secure_message_b64": _b64encode(sm.serialize()),
            }
            return web.json_response(envelope)
        except Exception:
            return web.json_response(payload)

    async def _handle_upload_fragment(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            file_id = str(msg["file_id"])
            fragment_index = int(msg["fragment_index"])
            fragment_hash = str(msg["fragment_hash"])
            file_merkle_root = str(msg["merkle_root"])
            blob = _b64decode(str(msg["fragment_blob_b64"]))
            owner_id = str(msg.get("owner_id") or "unknown")
            meta = self.storage.put_fragment(
                owner_id=owner_id,
                file_id=file_id,
                fragment_index=fragment_index,
                fragment_blob=blob,
                file_merkle_root_hex=file_merkle_root,
                fragment_hash_hex=fragment_hash,
            )
            return {"ok": True, "block_merkle_root_hex": meta.block_merkle_root_hex}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _handle_integrity_challenge(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fragment_hash = str(msg["fragment_hash"])
            block_index = int(msg["block_index"])
            nonce_b64 = str(msg["nonce_b64"])
            expected_root = str(msg["block_merkle_root_hex"])
            ch = PorChallenge(
                fragment_hash_hex=fragment_hash,
                block_index=block_index,
                nonce_b64=nonce_b64,
                expected_block_merkle_root_hex=expected_root,
            )
            resp = self.storage.answer_por_challenge(ch)
            return {"ok": True, "response": resp.__dict__}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _handle_get_fragment(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fragment_hash = str(msg["fragment_hash"])
            blob = self.storage.get_fragment(fragment_hash)
            return {"ok": True, "fragment_blob_b64": _b64encode(blob)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

