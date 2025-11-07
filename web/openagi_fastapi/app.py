from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import time
import base64
import hashlib
import secrets
import subprocess
from pathlib import Path

app = FastAPI(title="OpenAGI Gateway", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_SUMMARY = os.path.join(ROOT, "config", "archon_project_summary.json")
KEY_FILE = os.path.join(ROOT, "keys", "encryption.key")
IPFS_DIR = os.path.join(ROOT, "web", "data", "ipfs")
CHAT_DIR = os.path.join(ROOT, "web", "data", "chat_php")
os.makedirs(IPFS_DIR, exist_ok=True)
VERIFY_JS_PATH = Path(__file__).resolve().parent / 'verify_sig.js'
UI_DIR = (Path(__file__).resolve().parent.parent.parent / 'dapps' / 'secure-chat' / 'ui')
EVENT_SECRET = os.environ.get("OPENAGI_EVENT_SECRET", "openagi-dev-secret")


class CryptoPayload(BaseModel):
    payload: str
    room_id: str


@app.get("/status")
def status():
    st = {
        "name": "OpenAGI",
        "state": "desconocido",
        "timestamp": int(time.time()),
        "version": None,
        "last_update": None,
    }
    try:
        if os.path.exists(CONFIG_SUMMARY):
            with open(CONFIG_SUMMARY, "r", encoding="utf-8") as f:
                data = json.load(f)
                st["state"] = data.get("status") or data.get("state") or "operacional"
                st["version"] = data.get("version")
                st["last_update"] = data.get("last_update")
    except Exception:
        pass
    return {"ok": True, "status": st}


def _load_key() -> bytes:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            raw = f.read()
            return hashlib.sha256(raw).digest()
    # fallback key (dev only)
    return hashlib.sha256(b"openagi-dev-key").digest()


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return bytes(out)


@app.post("/crypto/encrypt")
def encrypt(body: CryptoPayload):
    key = _load_key()
    payload_bytes = body.payload.encode("utf-8")
    cipher_bytes = _xor_bytes(payload_bytes, key)
    cipher_b64 = base64.b64encode(cipher_bytes).decode("ascii")
    return {"ok": True, "ciphertext": cipher_b64, "enc": True}


@app.post("/crypto/decrypt")
def decrypt(body: CryptoPayload):
    key = _load_key()
    try:
        cipher_bytes = base64.b64decode(body.payload.encode("ascii"))
    except Exception:
        raise HTTPException(status_code=400, detail="ciphertext inválido")
    plain_bytes = _xor_bytes(cipher_bytes, key)
    return {"ok": True, "plaintext": plain_bytes.decode("utf-8", errors="replace")}


class IpfsUpload(BaseModel):
    content: str


@app.post("/ipfs/upload")
def ipfs_upload(body: IpfsUpload):
    # Simulación de IPFS: generar CID como sha1 y guardar contenido
    cid = hashlib.sha1(body.content.encode("utf-8")).hexdigest()
    path = os.path.join(IPFS_DIR, cid + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.content)
    return {"ok": True, "cid": cid}


@app.get("/ipfs/get")
def ipfs_get(cid: str):
    path = os.path.join(IPFS_DIR, cid + ".txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="CID no encontrado")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"ok": True, "content": content}


@app.get("/stats/chat")
def chat_stats():
    rooms_count = 0
    messages_count = 0
    encrypted_count = 0
    attachments_count = 0

    try:
        rooms_path = os.path.join(CHAT_DIR, "rooms.json")
        if os.path.exists(rooms_path):
            with open(rooms_path, "r", encoding="utf-8") as f:
                rooms = json.load(f) or []
                rooms_count = len(rooms)
    except Exception:
        pass

    try:
        if os.path.isdir(CHAT_DIR):
            for name in os.listdir(CHAT_DIR):
                if name.startswith("messages_") and name.endswith(".json"):
                    path = os.path.join(CHAT_DIR, name)
                    with open(path, "r", encoding="utf-8") as f:
                        msgs = json.load(f) or []
                        messages_count += len(msgs)
                        for m in msgs:
                            if m.get("enc"):
                                encrypted_count += 1
                            if (m.get("type") or "text") == "attachment":
                                attachments_count += 1
    except Exception:
        pass

    return {
        "ok": True,
        "stats": {
            "rooms": rooms_count,
            "messages": messages_count,
            "encrypted_messages": encrypted_count,
            "attachments": attachments_count,
        },
    }


def _is_valid_session(token: str) -> tuple[bool, str | None]:
    if not token:
        return (False, None)
    sess_path = os.path.join(CHAT_DIR, f"session_{token}.json")
    if not os.path.exists(sess_path):
        return (False, None)
    try:
        with open(sess_path, 'r', encoding='utf-8') as f:
            s = json.load(f)
    except Exception:
        return (False, None)
    if s.get("exp", 0) < time.time():
        try:
            os.remove(sess_path)
        except Exception:
            pass
        return (False, None)
    return (True, s.get("address"))

def _get_room(room_id: str) -> dict | None:
    try:
        rooms_path = os.path.join(CHAT_DIR, "rooms.json")
        if not os.path.exists(rooms_path):
            return None
        with open(rooms_path, "r", encoding="utf-8") as f:
            rooms = json.load(f) or []
        for r in rooms:
            if r.get("id") == room_id:
                if "access" not in r:
                    r["access"] = "open"
                return r
    except Exception:
        return None
    return None

def _is_member(room_id: str, address: str | None) -> bool:
    if not address:
        return False
    try:
        path = os.path.join(CHAT_DIR, f"members_{room_id}.json")
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            members = json.load(f) or []
        addr_l = address.lower()
        return addr_l in [m.lower() for m in members]
    except Exception:
        return False


@app.websocket("/ws/{room_id}")
async def ws_room(websocket: WebSocket, room_id: str):
    # Leer token de query para autenticar suscripción
    token = websocket.query_params.get('token')
    if token is None:
        ok, addr = False, None
    else:
        ok, addr = _is_valid_session(token)
    if not ok:
        # Cerrar con policy violation
        await websocket.close(code=1008)
        return
    # Verificar acceso a la sala
    room = _get_room(room_id)
    if not room:
        await websocket.close(code=1008)
        return
    if (room.get("access") or "open") == "restricted" and not _is_member(room_id, addr):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    conns = room_connections.setdefault(room_id, set())
    conns.add(websocket)
    try:
        while True:
            # Mantener la conexión; no esperamos mensajes del cliente
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        conns.discard(websocket)


class MessageEvent(BaseModel):
    room_id: str
    message: dict


@app.post("/events/message")
async def publish_message(evt: MessageEvent, server_secret: str | None = Header(None, alias="X-Server-Secret")):
    # ligero control de autenticación para publicación de eventos
    if (server_secret or "") != (EVENT_SECRET or ""):
        raise HTTPException(status_code=401, detail="unauthorized")
    room_id = evt.room_id
    payload = evt.message
    conns = room_connections.get(room_id, set())
    # enviar como texto JSON
    dead = []
    for ws in list(conns):
        try:
            await ws.send_json({"type": "message", "room_id": room_id, "message": payload})
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns.discard(ws)
    return {"ok": True, "delivered": len(conns)}
    
    
def ensure_verify_js():
    if not VERIFY_JS_PATH.exists():
        VERIFY_JS_PATH.write_text(
            """
const { verifyMessage } = require('ethers');
const msg = process.argv[2];
const sig = process.argv[3];
try {
  const addr = verifyMessage(msg, sig);
  process.stdout.write(addr);
} catch (e) {
  process.stderr.write('ERR:' + (e && e.message ? e.message : String(e)));
  process.exit(1);
}
            """.strip(), encoding='utf-8'
        )


def recover_address_node(message: str, signature: str) -> str | None:
    try:
        ensure_verify_js()
        cmd = ['node', str(VERIFY_JS_PATH), message, signature]
        proc = subprocess.run(cmd, cwd=str(UI_DIR), capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            addr = proc.stdout.strip()
            return addr
        return None
    except Exception:
        return None


@app.get('/auth/challenge')
def auth_challenge(address: str = Query(..., min_length=3)):
    addr = address.lower()
    nonce = secrets.token_urlsafe(16)
    ts = int(time.time())
    msg = f"OpenAGI Login\nAddress: {addr}\nNonce: {nonce}\nTs: {ts}"
    # store challenge in a simple file under CHAT_DIR for persistence
    chal_path = os.path.join(CHAT_DIR, f"challenge_{addr}.json")
    with open(chal_path, 'w', encoding='utf-8') as f:
        json.dump({"nonce": nonce, "ts": ts, "exp": ts + 300}, f)
    return {"ok": True, "message": msg, "nonce": nonce}


class AuthVerify(BaseModel):
    address: str
    signature: str
    message: str


@app.post('/auth/verify')
def auth_verify(payload: AuthVerify):
    addr = payload.address.lower()
    chal_path = os.path.join(CHAT_DIR, f"challenge_{addr}.json")
    if not os.path.exists(chal_path):
        return {"ok": False, "error": "no challenge"}
    try:
        with open(chal_path, 'r', encoding='utf-8') as f:
            ch = json.load(f)
    except Exception:
        ch = None
    if not ch:
        return {"ok": False, "error": "no challenge"}
    if ch.get("exp", 0) < time.time():
        return {"ok": False, "error": "challenge expired"}
    if f"Nonce: {ch['nonce']}" not in payload.message:
        return {"ok": False, "error": "invalid message"}
    rec = recover_address_node(payload.message, payload.signature)
    if not rec or rec.lower() != addr:
        return {"ok": False, "error": "invalid signature"}
    token = base64.urlsafe_b64encode(os.urandom(32)).decode('ascii').rstrip('=')
    sess_path = os.path.join(CHAT_DIR, f"session_{token}.json")
    with open(sess_path, 'w', encoding='utf-8') as f:
        json.dump({"address": addr, "exp": int(time.time()) + 86400}, f)
    # cleanup used challenge
    try:
        os.remove(chal_path)
    except Exception:
        pass
    return {"ok": True, "token": token, "address": addr}


@app.get('/auth/session/verify')
def auth_session_verify(token: str = Query(...)):
    sess_path = os.path.join(CHAT_DIR, f"session_{token}.json")
    if not os.path.exists(sess_path):
        return {"ok": False}
    try:
        with open(sess_path, 'r', encoding='utf-8') as f:
            s = json.load(f)
    except Exception:
        return {"ok": False}
    if s.get("exp", 0) < time.time():
        try:
            os.remove(sess_path)
        except Exception:
            pass
        return {"ok": False}
    return {"ok": True, "address": s.get("address")}


@app.post('/auth/logout')
def auth_logout(token: str = Query(...)):
    sess_path = os.path.join(CHAT_DIR, f"session_{token}.json")
    try:
        if os.path.exists(sess_path):
            os.remove(sess_path)
    except Exception:
        # Best-effort removal; ignore errors
        pass
    return {"ok": True}
# Registro simple de conexiones por sala
room_connections: dict[str, set] = {}