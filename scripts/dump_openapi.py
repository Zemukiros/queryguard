#!/usr/bin/env python3
"""Write the API's OpenAPI schema to a file, without a server, database or API key.

Usage:  uv run python scripts/dump_openapi.py web/openapi.json
Used by `npm run gen:api` (web/), which turns it into web/src/api/schema.ts.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from queryguard.api.app import create_app
from queryguard.api.settings import Settings


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        # A throwaway SQLite file: building the app opens its store. Nothing else
        # is touched -- the schema and LLM clients are created on first request.
        spec = create_app(Settings(db_path=Path(tmp) / "app.db")).openapi()
    Path(argv[0]).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {argv[0]} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
