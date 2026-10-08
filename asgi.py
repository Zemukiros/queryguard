"""Vercel's entrypoint for the `app` service (vercel.json: "entrypoint": "asgi:app").

Vercel resolves a `module:variable` entrypoint to a file relative to the
service root, so the app has to be reachable as a top-level module there.
The package lives under src/ (a src layout), and whether the build installs
it as a package is the platform's business: putting src/ on sys.path here
makes the import work either way. Locally, use `python -m queryguard.api`.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from queryguard.api.asgi import app  # noqa: E402

__all__ = ["app"]
