"""Serve the API, or export feedback.

    uv run python -m queryguard.api [--host 127.0.0.1] [--port 8000]
    uv run python -m queryguard.api export-feedback [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from queryguard.api.settings import Settings
from queryguard.api.store import FEEDBACK_CANDIDATES, Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m queryguard.api")
    sub = parser.add_subparsers(dest="command")
    export = sub.add_parser("export-feedback", help="write incorrect-feedback answers as golden candidates")
    export.add_argument("--out", type=Path, default=FEEDBACK_CANDIDATES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "export-feedback":
        path, count = Store(Settings.from_env().db_path).export_feedback_candidates(args.out)
        print(f"wrote {count} candidate(s) to {path}")
        return 0

    import uvicorn

    uvicorn.run("queryguard.api.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
