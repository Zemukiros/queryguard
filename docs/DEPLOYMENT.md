# Deployment

QueryGuard runs as one Vercel project (Hobby, $0) with two Vercel Services, a Neon Postgres (free) holding the
queried data, and an Upstash Redis (free) holding the API's own state. Production deploys from `main` through
Vercel's GitHub integration. No deploy token is stored anywhere.

```
browser ──► Vercel ──┬─ /v1/*, /healthz, /docs, /openapi.json ──► app  (FastAPI, Python 3.12, Fluid compute)
                     │                                             ├─► Neon     as queryguard_ro (SELECT only)
                     │                                             └─► Upstash  limits, spend, cache, history
                     └─ everything else ──────────────────────────► web  (Vite build, static)
```

## Pieces

| Piece | Where | Notes |
|---|---|---|
| `vercel.json` | repo root | Services `web` (`web/`, Vite) and `app` (repo root, `asgi:app`). Top-level rewrites route `/v1/*`, `/healthz`, `/docs`, `/openapi.json` to `app` and the rest to `web`. A service receives the original path. |
| `asgi.py` | repo root | Entrypoint shim: puts `src/` on `sys.path`, then exposes `queryguard.api.asgi:app`. Vercel resolves a `module:var` entrypoint to a file relative to the service root. |
| Schema cache | built in `app`'s build step | `introspect --refresh` with `DATABASE_URL_READONLY`; the owner URL never reaches Vercel. If the step fails, the API introspects with the read-only URL at first use. |
| App state | Upstash, `QUERYGUARD_REDIS_URL` | `RedisState`; keys prefixed `production:` / `preview:` from `VERCEL_ENV`. |
| Queried data | Neon, `DATABASE_URL_READONLY` | Seeded with `make neon-init` from `db/init/` (reads `.env.neon`, prints no secret). |
| Logs | Vercel function logs | Executions as JSON lines on stdout; the file ledger and calibration log are off (`VERCEL=1`). |

## Environment variables (Vercel → Settings → Environment Variables)

| Variable | Production | Preview | Why |
|---|---|---|---|
| `DATABASE_URL_READONLY` | Neon URL for `queryguard_ro` | same | the only database identity the API holds |
| `QUERYGUARD_REDIS_URL` | Upstash `rediss://…` | same | shared state; prefixes keep environments apart |
| `QUERYGUARD_LIVE` | `1` once verified, else `0` | `0` | kill switch: `0` = every question runs in demo mode |
| `QUERYGUARD_DAILY_SPEND_USD` | `0.50` | (default) | the ceiling; past it questions fall back to demo mode |
| `QUERYGUARD_DAILY_CALL_CAP` | `150` | (default) | backstop under the ceiling (~37 questions = ~130 calls) |
| `ANTHROPIC_API_KEY` | key from the `queryguard-demo` workspace | **never** | previews can never spend |

The Anthropic workspace has a $15/month spend limit: a hard stop that does not depend on this code.
Changing an environment variable takes effect on the **next deployment**: redeploy after every change.

## Launch order

1. Production env: `QUERYGUARD_DAILY_SPEND_USD=0.50`, `QUERYGUARD_DAILY_CALL_CAP=150`, `QUERYGUARD_LIVE=0`.
2. Production env: `ANTHROPIC_API_KEY` (Production only). With `LIVE=0` it is never used.
3. Squash-merge the PR into `main`. Vercel deploys production.
4. Verify production in demo mode: `/healthz` says `demo` / `switched_off` and a $0.50 ceiling; ask a question.
5. Production env: `QUERYGUARD_LIVE=1`, then redeploy production.
6. Verify live: `/healthz` says `live`; one question comes back `mode: live` with a cost; spend shows in the header.
7. Kill-switch drill: `QUERYGUARD_LIVE=0`, redeploy, confirm demo mode; then back to `1` and redeploy.

**Emergency stop:** set `QUERYGUARD_LIVE=0` and redeploy, or revoke the Anthropic key. Either is enough on its own.
A missing, expired or refused key (401/403) never shows up as an error: questions run in demo mode with
`mode_reason: "model_unavailable"`. A question that hits a refused key on its first call is re-run in demo mode
before anything reaches the browser. The instance then skips the live path for 5 minutes before it tries the key
again.

## Measured on preview deployments (demo mode)

| | Before lazy imports | After |
|---|---|---|
| `/healthz` after 15 min idle (cold) | 6.96 s | 3.29 s |
| `/healthz` after 30 min idle (cold) | 5.91 s | 2.36 s |
| `import asgi` locally (warm) | 1.12–1.15 s | 0.38–0.44 s |

- **Imports moved to the first question.** The lazy imports move pandas, sqlalchemy and the pipeline to the
  first question on a fresh instance. On the preview that added about 2 s before its first event. So the page
  now sends `POST /v1/warm` on load, fire and forget. It loads those modules and opens one read-only connection,
  which also wakes Neon. It makes no model call, spends nothing, isn't rate-limited and runs once per instance.
- **Other checks on the preview:**
  - **SSE:** events reach the browser in separate chunks, 0.12–0.18 s after the server sends them.
  - **Rate limit:** 14 concurrent questions against the 10/minute limit: exactly 10 admitted.
  - **Spoofing:** a spoofed `x-real-ip` was refused.
