#!/usr/bin/env bash
# Start the API and the Vite dev server together; Ctrl-C stops both.
#
#   scripts/dev.sh          real model: questions cost API calls (needs ANTHROPIC_API_KEY)
#   FAKE=1 scripts/dev.sh   demo mode: simulated model, $0, no key needed
#
# Open http://localhost:5173. Vite proxies /v1 and /healthz to the API on :8000.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${FAKE:-0}" == "1" ]]; then
  export QUERYGUARD_FAKE_LLM=1
  echo "demo mode: simulated model, \$0"
else
  echo "REAL model: every new question costs API calls (daily ceiling: \$${QUERYGUARD_DAILY_SPEND_USD:-1.00})"
fi

[[ -d web/node_modules ]] || (cd web && npm ci)

pids=()
cleanup() { trap - INT TERM EXIT; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup INT TERM EXIT

uv run python -m queryguard.api --host 127.0.0.1 --port 8000 & pids+=($!)
# Start the UI once the API answers, so the first page load does not see it offline.
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8000/healthz >/dev/null && break; sleep 0.5; done
(cd web && npx vite --host 127.0.0.1) & pids+=($!)

echo "open http://localhost:5173"
wait -n "${pids[@]}"
