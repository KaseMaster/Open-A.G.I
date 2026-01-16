import argparse
import os
import sys

from aiohttp import web


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--node-id", default="aegis-storage")
    p.add_argument("--security", default="HIGH")
    p.add_argument("--ui-dir", default=os.path.join(os.getcwd(), "ui"))
    args = p.parse_args()

    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    from aegis_storage import AegisStorageDApp
    from aegis_storage_dashboard_server import create_dashboard_app

    dapp = AegisStorageDApp(node_id=args.node_id, security_level=args.security)
    app = create_dashboard_app(dapp=dapp, ui_dir=args.ui_dir)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

