# syntax=docker/dockerfile:1
# QueryGuard API. Two stages: uv resolves the locked environment, then only the
# virtualenv, the package source and the golden set (demo mode answers from it)
# are copied into a slim runtime that runs as an unprivileged user.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so a source change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
# config.REPO_ROOT is found by walking up to pyproject.toml, so the editable
# install keeps the source layout: /app/pyproject.toml, /app/src, /app/.venv.
COPY --from=build /app /app
COPY evals/golden.yaml evals/golden.yaml
# data: API state (SQLite) · logs: LLM and execution logs · state: schema cache
RUN mkdir -p data logs state && chown app:app data logs state
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    QUERYGUARD_SCHEMA_CACHE=/app/state/schema_cache.json
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["python", "-m", "queryguard.api", "--host", "0.0.0.0", "--port", "8000"]
