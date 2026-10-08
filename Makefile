# QueryGuard developer commands. `make dev FAKE=1` is the $0 way to see the UI.
.PHONY: help dev api web gen-api test test-py test-web e2e lint check

FAKE ?= 0

help:
	@echo "make dev FAKE=1   API + web UI, simulated model, \$$0 (open http://localhost:5173)"
	@echo "make dev          API + web UI, real model (costs API calls)"
	@echo "make api          API only on :8000 (FAKE=1 for demo mode)"
	@echo "make web          Vite dev server only on :5173"
	@echo "make gen-api      regenerate web/src/api/schema.ts from the FastAPI schema"
	@echo "make test         pytest + vitest"
	@echo "make e2e          Playwright against demo mode (needs docker compose up -d)"
	@echo "make check        lint + typecheck + build + all tests"

dev:
	FAKE=$(FAKE) scripts/dev.sh

api:
	QUERYGUARD_FAKE_LLM=$(FAKE) uv run python -m queryguard.api --host 127.0.0.1 --port 8000

web:
	cd web && npx vite --host 127.0.0.1

gen-api:
	cd web && npm run gen:api

test-py:
	uv run pytest -q

test-web:
	cd web && npm test

test: test-py test-web

e2e:
	cd web && npm run e2e

lint:
	cd web && npm run lint && npm run typecheck

check: lint test e2e
	cd web && npm run build
