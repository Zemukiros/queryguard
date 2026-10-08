"""The ASGI app object for platforms that import one: Vercel's Python runtime
(vercel.json: services.app.entrypoint = "queryguard.api.asgi:app") or
`uvicorn queryguard.api.asgi:app`. Settings come from the environment."""

from queryguard.api.app import create_app

app = create_app()
