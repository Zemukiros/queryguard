"""HTTP API over the pipeline: one-shot and streamed queries, schema, history, feedback.

    uv run python -m queryguard.api                  # serve on :8000
    uv run python -m queryguard.api export-feedback  # incorrect answers -> evals/feedback_candidates.yaml

Endpoints (all bodies are Pydantic models, so /openapi.json is complete):

    POST /v1/query          QueryResult, after the whole pipeline has run
    POST /v1/query/stream   Server-Sent Events: `event: <stage>`, `data: <StageEvent JSON>`
    POST /v1/run[/stream]   user-supplied SQL for a question, through the same guardrail, executor,
                            validators, cache, rate limit and spend ceiling (events from `guardrails` on)
    GET  /v1/schema         the rendered prompt schema plus table/column metadata
    GET  /v1/history        the caller's recent questions, with outcome and confidence
    GET  /v1/history/{id}   one of the caller's past results, to reopen it
    POST /v1/feedback       {query_id, correct, note}
    GET  /healthz           liveness, scorer version and today's spend

Admission, in order, before any API call: a cache hit is answered at once
(cached=true, $0) and consumes no rate limit; otherwise the per-client rate
limit (429 + Retry-After); then the mode. A question runs live unless the
deployment is demo-only, live answers are switched off (QUERYGUARD_LIVE=0),
today's spend ceiling has no room, or today's call cap is nearly used -- then
it runs in demo mode (the simulated model, $0) and says why (`mode`,
`mode_reason`). Demo answers are never cached. See settings.py for every
limit. A streamed cache hit is a single `done` event.

The database boundary is unchanged: generated SQL still runs only as
`queryguard_ro` through the executor. The app's own state (limits, spend,
cache, history) sits behind queryguard.state.AppState.

Exception text from a stage is logged, not returned: a database or SDK error
message can carry hostnames and request details that do not belong in a public
response. Clients get the stage, the exception type and a short message.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from queryguard.api.limits import seconds_until_utc_midnight
from queryguard.api.settings import Settings
from queryguard.events import ErrorPayload, Mode, ModeReason, Out, QueryResult, StageEvent
from queryguard.llm.client import DEFAULT_MODEL, MAX_CALLS_PER_QUESTION, LLMClient, RequestCapExceeded, sdk_error_types
from queryguard.generate import Ambiguity, GeneratedSQL
from queryguard.text import normalize_question, normalize_sql
from queryguard.schema.introspect import DatabaseSchema, TableInfo, load_schema
from queryguard.state import AppState, Reservation, make_state
from queryguard.validation.backtranslate import VALIDATION_MODEL
from queryguard.validation.confidence import SCORER_VERSION

logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"
MAX_QUESTION_CHARS = 500
MAX_SQL_CHARS = 5000
USER_SQL_SELF_CONFIDENCE = 0.5
MAX_HISTORY = 100

# Outcomes worth replaying for free. A failed or errored run might succeed on a
# retry, and one with a validation error is missing a signal; neither is cached.
_CACHEABLE = {"answered", "clarification", "cannot_answer", "blocked", "refused"}

# Startup cost is a cold start on a serverless platform, so this module imports
# nothing heavy at load time: the pipeline (pandas, sqlalchemy, sqlparse) loads
# with the first question, the simulated model (yaml) with the first demo
# answer, and the Anthropic SDK only when a real client is built. /healthz and
# the OpenAPI schema need none of them. tests/test_platform.py holds that line.


def _public_message(exc: BaseException | None) -> bool:
    """Whether the exception's message is ours, written for the caller, and safe to return."""
    from queryguard.pipeline import QuestionBudgetExceeded  # loaded already: an error came from a run

    return isinstance(exc, (RequestCapExceeded, QuestionBudgetExceeded))


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


class RunRequest(BaseModel):
    """SQL to run for a question: an edited query, or a clarification's chosen reading."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    sql: str = Field(min_length=1, max_length=MAX_SQL_CHARS)
    source: Literal["user", "reading"] = Field(
        "user", description="reading: the SQL is one of a clarification's readings. Labels only; checks are identical."
    )

    @field_validator("question", "sql")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ErrorResponse(Out):
    """A stage raised. No QueryResult exists; nothing after `failed_stage` ran."""

    query_id: str
    failed_stage: str
    error_type: str
    message: str


class Detail(Out):
    """FastAPI's error body: 429, 503, 404 and 422."""

    detail: Any


class SchemaResponse(Out):
    schema_hash: str
    extracted_at: datetime
    rendered: str = Field(description="Exactly what the model is shown.")
    tables: list[TableInfo]


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=64)
    correct: bool
    note: str | None = Field(None, max_length=1000)


class FeedbackResponse(Out):
    query_id: str
    correct: bool
    note: str | None
    created_at: str


class HistoryFeedback(Out):
    correct: bool
    note: str | None
    created_at: str


class HistoryItem(Out):
    query_id: str
    created_at: str
    question: str
    outcome: str = Field(description="A QueryResult outcome, or 'error'.")
    confidence: float | None
    cached: bool
    cost_usd: float
    sql_source: str = Field(description="model; user or reading for /v1/run.")
    mode: Mode = "live"
    feedback: HistoryFeedback | None = None


class Budget(Out):
    spent_today_usd: float
    daily_ceiling_usd: float


class Health(Out):
    status: str
    version: str
    scorer_version: str
    fake_llm: bool = Field(description="True: a demo-only deployment; answers always come from llm/fake.py.")
    mode: Mode = Field(description="What a new question would run as right now.")
    mode_reason: ModeReason | None = Field(description="Why demo mode; None when live.")
    resets_in_s: int | None = Field(description="Seconds until the budget and call cap reset (00:00 UTC), "
                                                "when they are the reason for demo mode.")
    budget: Budget


# ------------------------------------------------------------------ helpers


def schema_hash(schema: DatabaseSchema) -> str:
    """Content hash; ignores when the schema was introspected."""
    return hashlib.sha256(schema.model_dump_json(exclude={"extracted_at"}).encode()).hexdigest()


def cache_key(question: str, schema: DatabaseSchema, sql: str | None = None) -> str:
    """A generated answer is keyed on the question; a /v1/run answer on the question and its SQL.

    The models and scorer are in the key: changing either changes the answer.
    """
    parts = ("query" if sql is None else "run", normalize_question(question), normalize_sql(sql or ""),
             schema_hash(schema), DEFAULT_MODEL, VALIDATION_MODEL, SCORER_VERSION)
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def user_answer(sql: str) -> GeneratedSQL:
    """User SQL as an answer, built without validation.

    GeneratedSQL's validator rejects anything that does not start with SELECT,
    which is right for model output but would turn a pasted DROP TABLE into a
    422 -- hiding the guardrail rejection the user should see. The guardrail
    and the read-only role are the gates; this only carries the text to them.
    Self-confidence is a neutral 0.5 (the calibrated weight on it is 0).
    """
    return GeneratedSQL.model_construct(
        sql=sql.strip(), explanation="SQL supplied by the user.", confidence=USER_SQL_SELF_CONFIDENCE,
        tables_used=[], columns_used=[], assumptions=[],
        ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]),
    )


def sse(event: StageEvent) -> str:
    return f"event: {event.stage}\ndata: {event.model_dump_json()}\n\n"


def _public_error(event: StageEvent, query_id: str) -> ErrorResponse:
    payload: ErrorPayload = event.payload
    exc = event.exception
    if _public_message(exc):
        message = payload.message
    elif isinstance(exc, sdk_error_types()):
        message = "the language model API request failed; try again shortly"
    else:
        message = "internal error; the details are in the server log"
    return ErrorResponse(
        query_id=query_id, failed_stage=payload.failed_stage, error_type=payload.error_type, message=message
    )


def _error_status(exc: BaseException | None) -> int:
    if _public_message(exc):
        return 503
    if isinstance(exc, sdk_error_types()):
        return 502
    return 500


# ---------------------------------------------------------------------- app


class _DropCacheWarning(logging.Filter):
    """The fake reports zero tokens, which LLMClient reads as an ignored cache marker."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "cache_control was set" not in record.getMessage()


@dataclass
class _Admission:
    """What admit() decided for one question."""

    client: str
    key: str
    cached: QueryResult | None = None
    reservation: Reservation | None = None
    mode: Mode = "live"
    reason: ModeReason | None = None


class _State:
    """What the routes share. LLM clients and the schema are created on first use."""

    def __init__(self, settings: Settings, schema: DatabaseSchema | None,
                 client: LLMClient | None, validation_client: LLMClient | None) -> None:
        self.settings = settings
        self.store: AppState = make_state(settings)
        self._schema = schema
        self._client = client
        self._validation_client = validation_client
        self._demo: tuple[LLMClient, LLMClient] | None = None
        self._lock = asyncio.Lock()
        self._health: tuple[float, tuple[Mode, ModeReason | None, float]] | None = None

    async def schema(self) -> DatabaseSchema:
        if self._schema is None:
            async with self._lock:
                if self._schema is None:
                    self._schema = await asyncio.to_thread(load_schema)
        return self._schema

    def clients(self) -> tuple[LLMClient, LLMClient]:
        # Real clients are capped and recorded by the state's call guard: with
        # Redis state, a daily call cap and spend ledger shared by every instance.
        if self._client is None:
            self._client = LLMClient(guard=self.store.call_guard())
        if self._validation_client is None:
            self._validation_client = LLMClient(model=VALIDATION_MODEL, guard=self.store.call_guard())
        return self._client, self._validation_client

    def demo_clients(self) -> tuple[LLMClient, LLMClient]:
        """The simulated model (llm/fake.py), behind a guard that never caps or counts spend."""
        if self._demo is None:
            client_logger = logging.getLogger("queryguard.llm.client")
            if not any(isinstance(f, _DropCacheWarning) for f in client_logger.filters):
                client_logger.addFilter(_DropCacheWarning())
            from queryguard.llm.fake import DemoFakeSDK

            fake, guard = DemoFakeSDK(), self.store.demo_call_guard()
            self._demo = (LLMClient(sdk_client=fake, guard=guard),
                          LLMClient(sdk_client=fake, model=VALIDATION_MODEL, guard=guard))
        return self._demo

    def blocked_mode(self) -> ModeReason | None:
        """A reason that applies before any budget check: a demo deployment or the kill switch."""
        if self.settings.fake_llm:
            return "demo_deployment"
        if not self.settings.live:
            return "switched_off"
        return None

    def call_cap_near(self) -> bool:
        """True when a whole question might not fit under today's call cap."""
        return self.store.calls_today() + MAX_CALLS_PER_QUESTION > self.settings.daily_call_cap

    def current_mode(self) -> tuple[Mode, ModeReason | None, float]:
        """(mode, reason, spent today) for /healthz, cached for 15 s to spare the state store."""
        now = time.monotonic()
        if self._health is not None and now - self._health[0] < 15:
            return self._health[1]
        spent = self.store.spent_today()
        reason = self.blocked_mode()
        if reason is None:
            if spent + self.settings.question_reserve_usd > self.settings.daily_spend_usd:
                reason = "budget"
            elif self.call_cap_near():
                reason = "call_cap"
        value: tuple[Mode, ModeReason | None, float] = ("demo" if reason else "live", reason, spent)
        self._health = (now, value)
        return value


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
        if settings.vercel:
            # Vercel's edge sets x-real-ip to the connecting client; a visitor
            # cannot supply it (unlike the first entries of x-forwarded-for).
            real_ip = request.headers.get("x-real-ip")
            if real_ip:
                return real_ip.strip()
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                # The last hop is the one the trusted proxy appended; earlier
                # entries are whatever the client chose to send.
                return forwarded.split(",")[-1].strip()
        return request.client.host if request.client else "unknown"

    async def admit(question: str, sql: str | None, request: Request, source: str = "user") -> _Admission:
        """Cache, rate limit, then mode. Raises 429 before any API call.

        A live admission holds a reservation against the spend ceiling, which
        run() releases. A demo admission holds none: it spends nothing.
        """
        client = await asyncio.to_thread(state.store.client_key, client_ip(request))
        key = cache_key(question, await state.schema(), sql)

        cached = await asyncio.to_thread(state.store.cache_get, key)
        if cached is not None:
            result = QueryResult.model_validate(cached).model_copy(
                update={"query_id": uuid.uuid4().hex, "cached": True, "question": question,
                        "cost_usd": 0.0, "n_calls": 0, "elapsed_ms": 0,
                        **({"sql_source": source} if sql is not None else {})}
            )
            await asyncio.to_thread(record, result, client)
            return _Admission(client, key, cached=result)

        retry_after = await asyncio.to_thread(state.store.rate_hit, client)
        if retry_after is not None:
            raise HTTPException(
                429,
                detail=f"rate limit: {settings.rate_per_minute}/minute and {settings.rate_per_day}/day per client",
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )

        reason = state.blocked_mode()
        if reason is not None:
            return _Admission(client, key, mode="demo", reason=reason)
        reservation, _ = await asyncio.to_thread(state.store.try_reserve)
        if reservation is None:
            return _Admission(client, key, mode="demo", reason="budget")
        if await asyncio.to_thread(state.call_cap_near):
            await asyncio.to_thread(state.store.release, reservation)
            return _Admission(client, key, mode="demo", reason="call_cap")
        return _Admission(client, key, reservation=reservation)

    def record(result: QueryResult, client: str) -> None:
        state.store.record(
            query_id=result.query_id, client=client, question=result.question, outcome=result.outcome,
            confidence=result.confidence, cached=result.cached, cost_usd=result.cost_usd,
            elapsed_ms=result.elapsed_ms, sql_source=result.sql_source, result=result.model_dump(mode="json"),
            mode=result.mode,
        )

    async def run(question: str, sql: str | None, admitted: _Admission, query_id: str,
                  source: str = "user") -> AsyncIterator[StageEvent]:
        """The pipeline's events, recorded and cached on the way out. Releases the reservation.

        sql None: generate (stream_question). Otherwise run that SQL (stream_answer),
        which is the same guardrail, executor and validators from `guardrails` on.
        Demo mode runs the same pipeline on the simulated model. An `error`
        event comes out already made public (see _public_error).
        """
        from queryguard.pipeline import stream_answer, stream_question

        client, key = admitted.client, admitted.key
        try:
            llm, validation_llm = state.clients() if admitted.mode == "live" else state.demo_clients()
            schema = await state.schema()
            if sql is None:
                stream = stream_question(question, client=llm, validation_client=validation_llm,
                                         schema=schema, query_id=query_id)
            else:
                stream = stream_answer(question, user_answer(sql), client=llm,
                                       validation_client=validation_llm, schema=schema, query_id=query_id)
            async for event in stream:
                if event.stage == "done":
                    result: QueryResult = event.payload
                    update: dict[str, Any] = {"mode": admitted.mode, "mode_reason": admitted.reason}
                    if sql is not None:
                        update["sql_source"] = source
                    result = result.model_copy(update=update)
                    event.payload = result
                    await asyncio.to_thread(record, result, client)
                    # Only live answers are cached: a cached answer is served as a
                    # real one, and a simulated answer must never be.
                    if admitted.mode == "live" and result.outcome in _CACHEABLE and not result.validation_errors:
                        await asyncio.to_thread(state.store.cache_put, key, result.model_dump(mode="json"))
                elif event.stage == "error":
                    logger.error("query %s failed in %s", query_id, event.payload.failed_stage,
                                 exc_info=event.exception)
                    error = _public_error(event, query_id)
                    await asyncio.to_thread(
                        state.store.record, query_id=query_id, client=client, question=question,
                        outcome="error", confidence=None, cached=False, cost_usd=0.0,
                        elapsed_ms=event.elapsed_ms, sql_source="model" if sql is None else source,
                        result=error.model_dump(), mode=admitted.mode,
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
            if admitted.reservation is not None:
                await asyncio.to_thread(state.store.release, admitted.reservation)

    async def answer_json(question: str, sql: str | None, request: Request, source: str = "user") -> Any:
        admitted = await admit(question, sql, request, source)
        if admitted.cached is not None:
            return admitted.cached
        query_id = uuid.uuid4().hex
        # aclosing: returning from inside `async for` leaves the generator
        # suspended, and its `finally` -- which releases the reservation --
        # would run only when garbage collection gets to it. On a serverless
        # instance frozen after the response, that can mean never, holding a
        # budget slot until the reservation expires.
        async with contextlib.aclosing(run(question, sql, admitted, query_id, source)) as events:
            async for event in events:
                if event.stage == "done":
                    return event.payload
                if event.stage == "error":
                    payload: ErrorPayload = event.payload
                    error = ErrorResponse(query_id=query_id, failed_stage=payload.failed_stage,
                                          error_type=payload.error_type, message=payload.message)
                    return JSONResponse(error.model_dump(), status_code=_error_status(event.exception))
        raise HTTPException(500, detail="pipeline ended without a result")

    async def answer_stream(question: str, sql: str | None, request: Request, source: str = "user") -> StreamingResponse:
        admitted = await admit(question, sql, request, source)

        async def events() -> AsyncIterator[str]:
            if admitted.cached is not None:
                yield sse(StageEvent(elapsed_ms=0, duration_ms=0, payload=admitted.cached))
                return
            # aclosing: a client that disconnects mid-stream closes this
            # generator; closing the inner one too releases the reservation now.
            async with contextlib.aclosing(run(question, sql, admitted, uuid.uuid4().hex, source)) as stream:
                async for event in stream:
                    yield sse(event)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-QueryGuard-Mode": admitted.mode},
        )

    # ------------------------------------------------------------- routes

    json_errors = {429: {"model": Detail}, 500: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}}
    stream_responses = {
        200: {"description": "text/event-stream. Each event's data is a StageEvent; the last is "
                             "`done` (payload: QueryResult) or `error` (payload: ErrorPayload).",
              "model": StageEvent},
        429: {"model": Detail}, 503: {"model": Detail},
    }

    @app.post("/v1/query", response_model=QueryResult, responses=json_errors)
    async def query(body: QueryRequest, request: Request) -> Any:
        return await answer_json(body.question, None, request)

    @app.post("/v1/query/stream", response_class=StreamingResponse, responses=stream_responses)
    async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        return await answer_stream(body.question, None, request)

    @app.post("/v1/run", response_model=QueryResult, responses=json_errors,
              summary="Run user-supplied SQL for a question, through the same checks")
    async def run_sql(body: RunRequest, request: Request) -> Any:
        return await answer_json(body.question, body.sql, request, body.source)

    @app.post("/v1/run/stream", response_class=StreamingResponse, responses=stream_responses,
              summary="Stream user-supplied SQL through the same checks (from `guardrails` on)")
    async def run_sql_stream(body: RunRequest, request: Request) -> StreamingResponse:
        return await answer_stream(body.question, body.sql, request, body.source)

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
                cost_usd=r["cost_usd"], sql_source=r["sql_source"], mode=r.get("mode") or "live",
                feedback=None if r["correct"] is None else HistoryFeedback(
                    correct=bool(r["correct"]), note=r["note"], created_at=r["feedback_at"]),
            )
            for r in rows
        ]

    @app.get("/v1/history/{query_id}", response_model=QueryResult, responses={404: {"model": Detail}})
    async def history_item(query_id: str, request: Request) -> Any:
        """A past result of the caller's, to reopen it. Failed questions have none."""
        result = await asyncio.to_thread(
            state.store.get_result, query_id, state.store.client_key(client_ip(request))
        )
        if result is None or result.get("stage") != "done":
            raise HTTPException(404, detail=f"no result for query {query_id!r}")
        return result

    @app.post("/v1/feedback", response_model=FeedbackResponse, responses={404: {"model": Detail}})
    async def feedback(body: FeedbackRequest) -> FeedbackResponse:
        if not await asyncio.to_thread(state.store.exists, body.query_id):
            raise HTTPException(404, detail=f"no query with id {body.query_id!r}")
        created_at = await asyncio.to_thread(state.store.save_feedback, body.query_id, body.correct, body.note)
        return FeedbackResponse(query_id=body.query_id, correct=body.correct, note=body.note, created_at=created_at)

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        """Liveness plus what a question would run as now. The web app calls this on
        load, which also starts a serverless instance while the visitor reads."""
        mode, reason, spent = await asyncio.to_thread(state.current_mode)
        return Health(
            status="ok", version=API_VERSION, scorer_version=SCORER_VERSION, fake_llm=settings.fake_llm,
            mode=mode, mode_reason=reason,
            resets_in_s=seconds_until_utc_midnight() if reason in ("budget", "call_cap") else None,
            budget=Budget(spent_today_usd=round(spent, 6), daily_ceiling_usd=settings.daily_spend_usd),
        )

    return app
