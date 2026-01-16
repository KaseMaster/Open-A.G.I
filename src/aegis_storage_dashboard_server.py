import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web


def create_dashboard_app(*, dapp: Any, ui_dir: str) -> web.Application:
    app = web.Application()
    ui_path = Path(ui_dir)

    async def index(_request: web.Request) -> web.Response:
        return web.FileResponse(ui_path / "dashboard.html")

    async def static_js(_request: web.Request) -> web.Response:
        return web.FileResponse(ui_path / "dashboard.js")

    async def status(_request: web.Request) -> web.Response:
        return web.json_response(dapp.get_audit_status())

    async def events(request: web.Request) -> web.Response:
        q = request.query.get("query", "")
        limit = int(request.query.get("limit", "200"))
        return web.json_response({"events": dapp.list_audit_events(query=q, limit=limit)})

    async def event_detail(request: web.Request) -> web.Response:
        event_id = request.match_info["eventId"]
        ev = dapp.get_audit_event(event_id)
        if ev is None:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(ev)

    async def event_proof(request: web.Request) -> web.Response:
        event_id = request.match_info["eventId"]
        proof = dapp.get_audit_proof(event_id)
        if proof is None:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(proof)

    async def verify(request: web.Request) -> web.Response:
        body = await request.read()
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            return web.json_response({"valid": False, "reason": "invalid_json"}, status=400)
        return web.json_response(dapp.verify_audit_proof(data))

    app.router.add_get("/", index)
    app.router.add_get("/dashboard.js", static_js)
    app.router.add_get("/api/audit/status", status)
    app.router.add_get("/api/audit/events", events)
    app.router.add_get("/api/audit/events/{eventId}", event_detail)
    app.router.add_get("/api/audit/events/{eventId}/proof", event_proof)
    app.router.add_post("/api/audit/verify", verify)
    return app


async def run_dashboard(*, dapp: Any, ui_dir: str, host: str = "127.0.0.1", port: int = 8099) -> None:
    app = create_dashboard_app(dapp=dapp, ui_dir=ui_dir)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    while True:
        await asyncio.sleep(3600)

