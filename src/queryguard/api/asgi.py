"""The ASGI app object for platforms that import one: Vercel's Python runtime
(through the root-level asgi.py shim that vercel.json names) or
`uvicorn queryguard.api.asgi:app`. Settings come from the environment."""

from queryguard.api.app import create_app

app = create_app()
