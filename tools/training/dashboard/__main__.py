"""Start the local RM Vision RL dashboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.config import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="bind address (default: dashboard.json or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="bind port (default: dashboard.json or 8765)")
    args = parser.parse_args()
    try:
        import uvicorn
        from tools.training.dashboard.server import create_app
    except RuntimeError as error:
        parser.error(str(error))
    except ImportError as error:
        parser.error(f"Web dependencies are missing; install requirements-web.txt: {error}")
    settings_path = ROOT / "config" / "tools" / "dashboard.json"
    try:
        settings = json.loads(settings_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        parser.error(f"invalid {settings_path}: {error}")
    host = args.host or settings.get("host", "127.0.0.1")
    port = args.port or int(settings.get("port", 8765))
    uvicorn.run(create_app(ROOT), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
