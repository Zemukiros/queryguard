"""The pipeline as a sequence of typed stage events.

`pipeline.stream_question` yields one StageEvent as each stage *finishes*,
carrying what that stage produced. The order for each kind of question:

    answered      generating > guardrails > executing > sanity
                  > backtranslate > agreement > confidence > done
    clarification generating > clarification > done
    cannot answer generating > done
    blocked       generating > guardrails > confidence > done
    db failure    generating > guardrails > executing > sanity > confidence > done
    exception     ... > error            (no done; nothing after it)

backtranslate and agreement appear only when validation ran, which needs a
query that executed. agreement is always emitted when validation ran; for a
trivial query it says it was skipped, so a client can rely on the sequence.

Every payload is a Pydantic model so the API's OpenAPI schema describes it
exactly; `done` carries the whole QueryResult, so a client that only wants the
answer can ignore everything before it. The Python-side outcome (PipelineResult
and friends, with its DataFrame) rides along on the event as a private
attribute, which is how the synchronous run_question / run_answer read the same
stream instead of duplicating it.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Annotated, Any, Literal, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from queryguard.generate import Interpretation

class Out(BaseModel):
    """A response model. Fields with defaults are always sent, so the schema marks them required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


Stage = Literal[
    "generating", "clarification", "guardrails", "executing", "sanity",
    "backtranslate", "agreement", "confidence", "done", "error",
]

Outcome = Literal["answered", "clarification", "cannot_answer", "blocked", "refused", "failed"]


class GeneratingPayload(Out):
    stage: Literal["generating"] = "generating"
    kind: Literal["sql", "clarification", "cannot_answer"]
    sql: str | None = None
    explanation: str
    self_confidence: float
    assumptions: list[str] = Field(default_factory=list)
    tables_used: list[str] = Field(default_factory=list)
    cost_usd: float
    latency_ms: int


class ClarificationPayload(Out):
    stage: Literal["clarification"] = "clarification"
    interpretations: list[Interpretation]


class GuardrailsPayload(Out):
    stage: Literal["guardrails"] = "guardrails"
    allowed: bool
    rule: str | None = None
    reason: str | None = None
    rewritten_sql: str | None = None
    sql_to_execute: str | None = None


class ColumnModel(Out):
    name: str
    dtype: str


class ExecutingPayload(Out):
    stage: Literal["executing"] = "executing"
    outcome: Literal["ok", "refused", "failed"]
    row_count: int
    truncated: bool
    execution_ms: int
    columns: list[ColumnModel] = Field(default_factory=list)
    estimated_rows: float | None = None
    reason: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    sqlstate: str | None = None


class SanityFlagModel(Out):
    check: str
    severity: Literal["fail", "warn", "info"]
    explanation: str
    column: str | None = None


class SanityPayload(Out):
    stage: Literal["sanity"] = "sanity"
    flags: list[SanityFlagModel]


class BacktranslatePayload(Out):
    stage: Literal["backtranslate"] = "backtranslate"
    back_translation: str | None = None
    alignment: float | None = None
    discrepancies: list[str] = Field(default_factory=list)
    error: str | None = None


class AgreementPayload(Out):
    stage: Literal["agreement"] = "agreement"
    ran: bool
    outcome: Literal["agree", "disagree", "incomparable"] | None = None
    explanation: str | None = None
    second_sql: str | None = None
    skipped_reason: str | None = None
    error: str | None = None


class Contribution(Out):
    """One term of the confidence logit: weight x value. The terms sum to `logit`."""

    feature: str
    label: str
    value: float
    weight: float
    contribution: float


Band = Literal["high", "medium", "low"]


class ConfidencePayload(Out):
    stage: Literal["confidence"] = "confidence"
    confidence: float
    band: Band
    logit: float | None = Field(None, description="None when nothing executed (the score is then 0).")
    contributions: list[Contribution] = Field(default_factory=list)
    breakdown: dict[str, float]
    scorer_version: str


class QueryResult(Out):
    """Everything a question produced. The `done` payload, and the body of POST /v1/query."""

    stage: Literal["done"] = "done"
    query_id: str
    question: str
    outcome: Outcome
    cached: bool = False
    sql_source: Literal["model", "user", "reading"] = Field(
        "model", description="user: SQL supplied to /v1/run; reading: a clarification's reading, run via /v1/run."
    )

    sql: str | None = Field(None, description="The SQL the model wrote.")
    executed_sql: str | None = Field(None, description="What ran: the guardrail may have added a LIMIT.")
    explanation: str | None = None
    assumptions: list[str] = Field(default_factory=list)

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list, description="JSON-safe cells, in column order.")
    row_count: int = 0
    truncated: bool = False
    execution_ms: int | None = None

    guardrail_rule: str | None = None
    guardrail_reason: str | None = None
    execution_error: str | None = None

    sanity: list[SanityFlagModel] = Field(default_factory=list)
    back_translation: str | None = None
    alignment: float | None = None
    discrepancies: list[str] = Field(default_factory=list)
    agreement: Literal["agree", "disagree", "incomparable"] | None = None
    agreement_explanation: str | None = None
    second_sql: str | None = Field(None, description="The independently written query agreement compared.")
    validation_errors: list[str] = Field(default_factory=list)

    confidence: float | None = None
    confidence_band: Band | None = None
    confidence_logit: float | None = None
    contributions: list[Contribution] = Field(default_factory=list)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    scorer_version: str | None = None

    interpretations: list[Interpretation] = Field(default_factory=list)
    cannot_answer_reason: str | None = None

    n_calls: int = 0
    cost_usd: float = 0.0
    elapsed_ms: int = 0


class ErrorPayload(Out):
    stage: Literal["error"] = "error"
    failed_stage: Stage
    error_type: str
    message: str


Payload = Annotated[
    Union[
        GeneratingPayload, ClarificationPayload, GuardrailsPayload, ExecutingPayload,
        SanityPayload, BacktranslatePayload, AgreementPayload, ConfidencePayload,
        QueryResult, ErrorPayload,
    ],
    Field(discriminator="stage"),
]


class StageEvent(Out):
    """One finished stage. On the wire: SSE `event: <stage>`, `data: <this as JSON>`."""

    elapsed_ms: int = Field(description="Since the question started.")
    duration_ms: int = Field(description="This stage alone.")
    payload: Payload

    # Python-side only: the typed outcome on `done`, the exception on `error`.
    _outcome: Any = PrivateAttr(default=None)
    _exception: BaseException | None = PrivateAttr(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stage(self) -> Stage:
        return self.payload.stage

    @property
    def result(self) -> Any:
        """On `done`: the PipelineResult, ClarificationNeeded or CannotAnswer."""
        return self._outcome

    @property
    def exception(self) -> BaseException | None:
        """On `error`: what was raised, so a synchronous caller can re-raise it."""
        return self._exception


def json_cell(value: Any) -> Any:
    """A result cell as JSON: Decimal to float, times to ISO strings, NaN/NaT to null.

    Intervals become their text form ("3 days 04:00:00"), not a bare number of
    seconds, so the unit is never lost. Arrays (array_agg) recurse.
    """
    if isinstance(value, list | tuple):
        return [json_cell(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, datetime | date | time):  # pd.Timestamp is a datetime
        return value.isoformat()
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value).hex()
    return str(value)
