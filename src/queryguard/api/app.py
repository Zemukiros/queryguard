"""HTTP API over the pipeline: one-shot and streamed queries, schema, history, feedback.

    uv run python -m queryguard.api                  # serve on :8000
    uv run python -m queryguard.api export-feedback  # incorrect answers -> evals/feedback_candidates.yaml

Endpoints (all bodies are Pydantic models, so /openapi.json is complete):

    POST /v1/query          QueryResult, after the whole pipeline has run
    POST /v1/query/stream   Server-Sent Events: `event: <stage>`, `data: <StageEvent JSON>`
    GET  /v1/schema         the rendered prompt schema plus table/column metadata
    GET  /v1/history        the caller's recent questions, with outcome and confidence
    POST /v1/feedback       {query_id, correct, note}
    GET  /healthz           liveness, scorer version and today's spend

Admission, in order, before any API call: a cache hit is answered at once
(cached=true, $0) and consumes no rate limit; otherwise the per-client rate
limit (429 + Retry-After), then the daily spend ceiling (503 + Retry-After).
See settings.py for every limit. A streamed cache hit is a single `done` event.

The database boundary is unchanged: generated SQL still runs only as
`queryguard_ro` through the executor. The app's own state is SQLite (store.py).

Exception text from a stage is logged, not returned: a database or SDK error
message can carry hostnames and request details that do not belong in a public
response. Clients get the stage, the exception type and a short message.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import unicodedata
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from queryguard.api.limits import RateLimiter, SpendCeiling, seconds_until_utc_midnight, spent_today_usd
from queryguard.api.settings import Settings
from queryguard.api.store import Store
from queryguard.events import ErrorPayload, QueryResult, StageEvent
from queryguard.llm.client import DEFAULT_MODEL, LLMClient, RequestCapExceeded
from queryguard.pipeline import QuestionBudgetExceeded, stream_question
from queryguard.schema.introspect import DatabaseSchema, TableInfo, load_schema
from queryguard.validation.backtranslate import VALIDATION_MODEL
from queryguard.validation.confidence import SCORER_VERSION

logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"
MAX_QUESTION_CHARS = 500
MAX_HISTORY = 100

# Outcomes worth replaying for free. A failed or errored run might succeed on a
# retry, and one with a validation error is missing a signal; neither is cached.
_CACHEABLE = {"answered", "clarification", "cannot_answer", "blocked", "refused"}

# Exceptions whose message is ours, written for the caller, and safe to return.
_PUBLIC_MESSAGE = (RequestCapExceeded, QuestionBudgetExceeded)


# ------------------------------------------------------------------ bodies


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question is blank")
        return value


class ErrorResponse(BaseModel):
    """A stage raised. No QueryResult exists; nothing after `failed_stage` ran."""

    query_id: str
    failed_stage: str
    error_type: str
    message: str


class Detail(BaseModel):
    """FastAPI's error body: 429, 503, 404 and 422."""

    detail: Any


class SchemaResponse(BaseModel):
    schema_hash: str
    extracted_at: datetime
    rendered: str = Field(description="Exactly what the model is shown.")
    tables: list[TableInfo]


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=64)
    correct: bool
    note: str | None = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    query_id: str
    correct: bool
    note: str | None
    created_at: str


class HistoryFeedback(BaseModel):
    correct: bool
    note: str | None
    created_at: str


class HistoryItem(BaseModel):
    query_id: str
    created_at: str
    question: str
    outcome: str = Field(description="A QueryResult outcome, or 'error'.")
    confidence: float | None
    cached: bool
    cost_usd: float
    feedback: HistoryFeedback | None = None


class Budget(BaseModel):
    spent_today_usd: float
    daily_ceiling_usd: float


class Health(BaseModel):
    status: str
    version: str
    scorer_version: str
    budget: Budget


# ------------------------------------------------------------------ helpers


def normalize_question(question: str) -> str:
    """Case, spacing and trailing punctuation do not change what is asked."""
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(text.split()).rstrip(" ?.!")


def schema_hash(schema: DatabaseSchema) -> str:
    """Content hash; ignores when the schema was introspected."""
    return hashlib.sha256(schema.model_dump_json(exclude={"extracted_at"}).encode()).hexdigest()


def cache_key(question: str, schema: DatabaseSchema) -> str:
    # The models and scorer are in the key: changing either changes the answer.
    parts = (normalize_question(question), schema_hash(schema), DEFAULT_MODEL, VALIDATION_MODEL, SCORER_VERSION)
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def sse(event: StageEvent) -> str:
    return f"event: {event.stage}\ndata: {event.model_dump_json()}\n\n"


def _public_error(event: StageEvent, query_id: str) -> ErrorResponse:
    payload: ErrorPayload = event.payload
    exc = event.exception
    if isinstance(exc, _PUBLIC_MESSAGE):
        message = payload.message
    elif isinstance(exc, anthropic.APIError):
        message = "the language model API request failed; try again shortly"
    else:
        message = "internal error; the details are in the server log"
    return ErrorResponse(
        query_id=query_id, failed_stage=payload.failed_stage, error_type=payload.error_type, message=message
    )


def _error_status(exc: BaseException | None) -> int:
    if isinstance(exc, _PUBLIC_MESSAGE):
        return 503
    if isinstance(exc, anthropic.APIError):
        return 502
    return 500


# ---------------------------------------------------------------------- app


class _State:
    """What the routes share. LLM clients and the schema are created on first use."""

    def __init__(self, settings: Settings, schema: DatabaseSchema | None,
                 client: LLMClient | None, validation_client: LLMClient | None) -> None:
        self.settings = settings
        self.store = Store(settings.db_path)
        self.limiter = RateLimiter(settings.rate_per_minute, settings.rate_per_day)
        self.ceiling = SpendCeiling(settings.daily_spend_usd, settings.question_reserve_usd)
        self._schema = schema
        self._client = client
        self._validation_client = validation_client
        self._lock = asyncio.Lock()

    async def schema(self) -> DatabaseSchema:
        if self._schema is None:
            async with self._lock:
                if self._schema is None:
                    self._schema = await asyncio.to_thread(load_schema)
        return self._schema

    def clients(self) -> tuple[LLMClient, LLMClient]:
        if self._client is None:
            self._client = LLMClient()
        if self._validation_client is None:
            self._validation_client = LLMClient(model=VALIDATION_MODEL)
        return self._client, self._validation_client


def create_app(
    settings: Settings | None = None,
    *,
    schema: DatabaseSchema | None = None,
    client: LLMClient | None = None,
    validation_client: LLMClient | None = None,
) -> FastAPI:
    """Build the app. Tests inject a schema and fake-backed clients; production passes nothing."""
    settings = settings or Settings.from_env()
    state = _State(settings, schema, client, validation_client)

    app = FastAPI(
        title="QueryGuard",
        version=API_VERSION,
        summary="Text-to-SQL behind a read-only database role, with hallucination detection.",
    )
    app.state.qg = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Retry-After"],
    )

    def client_ip(request: Request) -> str:
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                # The last hop is the one the trusted proxy appended; earlier
                # entries are whatever the client chose to send.
                return forwarded.split(",")[-1].strip()
        return request.client.host if request.client else "unknown"

    async def admit(body: QueryRequest, request: Request) -> tuple[str, str, QueryResult | None]:
        """(client key, cache key, cached result or None). Raises 429/503 before any API call."""
        client = state.store.client_key(client_ip(request))
        key = cache_key(body.question, await state.schema())

        cached = await asyncio.to_thread(state.store.cache_get, key)
        if cached is not None:
            result = QueryResult.model_validate(cached).model_copy(
                update={"query_id": uuid.uuid4().hex, "cached": True, "question": body.question,
                        "cost_usd": 0.0, "n_calls": 0, "elapsed_ms": 0}
            )
            await asyncio.to_thread(
                state.store.record, query_id=result.query_id, client=client, question=body.question,
                outcome=result.outcome, confidence=result.confidence, cached=True, cost_usd=0.0,
                elapsed_ms=0, result=result.model_dump(mode="json"),
            )
            return client, key, result

        retry_after = state.limiter.hit(client)
        if retry_after is not None:
            raise HTTPException(
                429,
                detail=f"rate limit: {settings.rate_per_minute}/minute and {settings.rate_per_day}/day per client",
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )
        admitted, spent = await asyncio.to_thread(state.ceiling.try_admit)
        if not admitted:
            raise HTTPException(
                503,
                detail=(
                    f"the public demo's daily budget of ${settings.daily_spend_usd:.2f} is used up "
                    f"(${spent:.2f} spent today, UTC); it resets at 00:00 UTC. Cached questions still work."
                ),
                headers={"Retry-After": str(seconds_until_utc_midnight())},
            )
        return client, key, None

    async def run(question: str, client: str, key: str, query_id: str) -> AsyncIterator[StageEvent]:
        """The pipeline's events, recorded and cached on the way out. Releases the spend reserve.

        An `error` event comes out already made public (see _public_error).
        """
        try:
            llm, validation_llm = state.clients()
            async for event in stream_question(
                question, client=llm, validation_client=validation_llm, schema=await state.schema(),
                query_id=query_id,
            ):
                if event.stage == "done":
                    result: QueryResult = event.payload
                    dumped = result.model_dump(mode="json")
                    await asyncio.to_thread(
                        state.store.record, query_id=result.query_id, client=client, question=question,
                        outcome=result.outcome, confidence=result.confidence, cached=False,
                        cost_usd=result.cost_usd, elapsed_ms=result.elapsed_ms, result=dumped,
                    )
                    if result.outcome in _CACHEABLE and not result.validation_errors:
                        await asyncio.to_thread(state.store.cache_put, key, dumped)
                elif event.stage == "error":
                    logger.error("query %s failed in %s", query_id, event.payload.failed_stage,
                                 exc_info=event.exception)
                    error = _public_error(event, query_id)
                    await asyncio.to_thread(
                        state.store.record, query_id=query_id, client=client, question=question,
                        outcome="error", confidence=None, cached=False, cost_usd=0.0,
                        elapsed_ms=event.elapsed_ms, result=error.model_dump(),
                    )
                    public = StageEvent(
                        elapsed_ms=event.elapsed_ms, duration_ms=event.duration_ms,
                        payload=ErrorPayload(failed_stage=error.failed_stage, error_type=error.error_type,
                                             message=error.message),
                    )
                    public._exception = event.exception
                    event = public
                yield event
        finally:
            state.ceiling.release()

    # ------------------------------------------------------------- routes

    @app.post(
        "/v1/query",
        response_model=QueryResult,
        responses={429: {"model": Detail}, 500: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}},
    )
    async def query(body: QueryRequest, request: Request) -> Any:
        client, key, cached = await admit(body, request)
        if cached is not None:
            return cached
        query_id = uuid.uuid4().hex
        async for event in run(body.question, client, key, query_id):
            if event.stage == "done":
                return event.payload
            if event.stage == "error":
                payload: ErrorPayload = event.payload
                error = ErrorResponse(query_id=query_id, failed_stage=payload.failed_stage,
                                      error_type=payload.error_type, message=payload.message)
                return JSONResponse(error.model_dump(), status_code=_error_status(event.exception))
        raise HTTPException(500, detail="pipeline ended without a result")

    @app.post(
        "/v1/query/stream",
        response_class=StreamingResponse,
        responses={
            200: {"description": "text/event-stream. Each event's data is a StageEvent; the last is "
                                 "`done` (payload: QueryResult) or `error` (payload: ErrorPayload).",
                  "model": StageEvent},
            429: {"model": Detail}, 503: {"model": Detail},
        },
    )
    async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        client, key, cached = await admit(body, request)

        async def events() -> AsyncIterator[str]:
            if cached is not None:
                yield sse(StageEvent(elapsed_ms=0, duration_ms=0, payload=cached))
                return
            async for event in run(body.question, client, key, uuid.uuid4().hex):
                yield sse(event)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/schema", response_model=SchemaResponse)
    async def get_schema() -> SchemaResponse:
        schema = await state.schema()
        return SchemaResponse(
            schema_hash=schema_hash(schema), extracted_at=schema.extracted_at,
            rendered=schema.render_for_prompt(), tables=schema.tables,
        )

    @app.get("/v1/history", response_model=list[HistoryItem])
    async def history(request: Request, limit: int = Query(20, ge=1, le=MAX_HISTORY)) -> list[HistoryItem]:
        rows = await asyncio.to_thread(state.store.history, state.store.client_key(client_ip(request)), limit)
        return [
            HistoryItem(
                query_id=r["query_id"], created_at=r["created_at"], question=r["question"],
                outcome=r["outcome"], confidence=r["confidence"], cached=bool(r["cached"]),
                cost_usd=r["cost_usd"],
                feedback=None if r["correct"] is None else HistoryFeedback(
                    correct=bool(r["correct"]), note=r["note"], created_at=r["feedback_at"]),
            )
            for r in rows
        ]

    @app.post("/v1/feedback", response_model=FeedbackResponse, responses={404: {"model": Detail}})
    async def feedback(body: FeedbackRequest) -> FeedbackResponse:
        if not await asyncio.to_thread(state.store.exists, body.query_id):
            raise HTTPException(404, detail=f"no query with id {body.query_id!r}")
        created_at = await asyncio.to_thread(state.store.save_feedback, body.query_id, body.correct, body.note)
        return FeedbackResponse(query_id=body.query_id, correct=body.correct, note=body.note, created_at=created_at)

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        spent = await asyncio.to_thread(spent_today_usd)
        return Health(
            status="ok", version=API_VERSION, scorer_version=SCORER_VERSION,
            budget=Budget(spent_today_usd=round(spent, 6), daily_ceiling_usd=settings.daily_spend_usd),
        )

    return app
