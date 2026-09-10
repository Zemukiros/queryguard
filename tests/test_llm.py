"""Tests for the prompt constructor and LLM client.

Every test here injects a fake SDK client. Nothing in this file constructs a
real anthropic.Anthropic, so the suite cannot spend money or require an API key.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from queryguard import generate as generate_mod
from queryguard.generate import (
    Ambiguity,
    ClarificationNeeded,
    GeneratedSQL,
    Interpretation,
    _render_clarification,
    generate_sql_with_stats,
)
from queryguard.llm import client as client_mod
from queryguard.llm.client import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING,
    CallResult,
    LLMClient,
    RequestCapExceeded,
    estimate_cost_usd,
    request_count,
    reset_request_count,
    running_total_usd,
)
from queryguard.llm.prompt import build_system_blocks
from queryguard.schema.introspect import ColumnInfo, DatabaseSchema, TableInfo

REQUIRED_LOG_FIELDS = {
    "timestamp",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "estimated_cost_usd",
    "latency_ms",
    "prompt_sha256",
}


@pytest.fixture(autouse=True)
def _isolate_request_counter():
    """The cap counter is process-global, so it must not leak between tests."""
    reset_request_count()
    yield
    reset_request_count()


class FakeUsage:
    def __init__(self, inp=1000, out=200, write=0, read=0):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_creation_input_tokens = write
        self.cache_read_input_tokens = read


class FakeMessage:
    def __init__(self, parsed, usage):
        self.parsed_output = parsed
        self.usage = usage


class FakeMessages:
    def __init__(self, parsed, usage):
        self._parsed = parsed
        self._usage = usage
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage(self._parsed, self._usage)


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic. Records calls, never uses the network."""

    def __init__(self, parsed=None, usage=None):
        self.messages = FakeMessages(parsed or _sample_answer(), usage or FakeUsage())

    @property
    def calls(self):
        return self.messages.calls


def _sample_answer() -> GeneratedSQL:
    return GeneratedSQL(
        sql="SELECT count(*) FROM orders WHERE status = 'cancelled';",
        explanation="Counts cancelled orders.",
        confidence=0.95,
        tables_used=["orders"],
        columns_used=["orders.status"],
        assumptions=[],
        ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]),
    )


def _ambiguous_answer() -> GeneratedSQL:
    """An ambiguous response: no top-level SQL, two competing readings."""
    return GeneratedSQL(
        sql="",
        explanation="'Revenue' can be gross or net of refunds.",
        confidence=0.0,
        tables_used=["orders", "refunds"],
        columns_used=["orders.total_amount", "refunds.amount"],
        assumptions=[],
        ambiguity=Ambiguity(
            is_ambiguous=True,
            interpretations=[
                Interpretation(
                    label="gross_revenue",
                    sql="SELECT sum(o.total_amount) AS revenue FROM orders AS o;",
                    explanation="Sums order totals, ignoring refunds.",
                ),
                Interpretation(
                    label="net_of_refunds",
                    sql=(
                        "SELECT sum(o.total_amount) - coalesce(sum(r.amount), 0) AS revenue\n"
                        "FROM orders AS o\n"
                        "LEFT JOIN refunds AS r ON r.order_id = o.order_id;"
                    ),
                    explanation="Gross less refunds issued against those orders.",
                ),
            ],
        ),
    )


def _synthetic_schema() -> DatabaseSchema:
    return DatabaseSchema(
        extracted_at=datetime.now(timezone.utc),
        tables=[
            TableInfo(
                name="widgets",
                row_count=3,
                comment="Test table.",
                columns=[
                    ColumnInfo(
                        name="widget_id", sql_type="integer", nullable=False, is_primary_key=True
                    ),
                    ColumnInfo(
                        name="kind",
                        sql_type="text",
                        nullable=False,
                        enum_values=["alpha", "beta"],
                        distinct_count=2,
                    ),
                ],
            )
        ],
    )


# ------------------------------------------------------------------- prompt


def test_cache_control_is_on_exactly_one_block_and_it_is_the_last() -> None:
    blocks = build_system_blocks(_synthetic_schema())

    assert len(blocks) == 3
    marked = [i for i, block in enumerate(blocks) if "cache_control" in block]
    assert marked == [2], "exactly one breakpoint, on the final block"
    assert blocks[2]["cache_control"] == {"type": "ephemeral"}


def test_rendered_schema_appears_verbatim_in_the_system_prompt() -> None:
    schema = _synthetic_schema()
    rendered = schema.render_for_prompt()

    blocks = build_system_blocks(schema)

    assert rendered in blocks[1]["text"], "schema block must carry the exact rendering"
    assert "widgets (3 rows)" in blocks[1]["text"]
    assert "∈ {alpha, beta}" in blocks[1]["text"]


def test_system_prompt_is_stable_across_questions() -> None:
    """The cached prefix must not vary, or every call is a cache miss."""
    schema = _synthetic_schema()
    assert build_system_blocks(schema) == build_system_blocks(schema)


# --------------------------------------------------------------- request cap


def test_request_cap_raises_without_calling_the_api(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUERYGUARD_MAX_REQUESTS", "2")
    fake = FakeAnthropic()
    llm = LLMClient(sdk_client=fake)
    blocks = build_system_blocks(_synthetic_schema())
    log = tmp_path / "llm_calls.jsonl"

    for _ in range(2):
        llm.complete(blocks, "Q: hi", output_format=GeneratedSQL, log=log)
    assert len(fake.calls) == 2
    assert request_count() == 2

    with pytest.raises(RequestCapExceeded):
        llm.complete(blocks, "Q: hi", output_format=GeneratedSQL, log=log)

    # The guard must stop the request, not merely report it afterwards.
    assert len(fake.calls) == 2, "API was called despite the cap"


def test_cap_is_not_reset_by_constructing_a_new_client(monkeypatch, tmp_path) -> None:
    """A runaway loop's natural move is a fresh client; that must not help."""
    monkeypatch.setenv("QUERYGUARD_MAX_REQUESTS", "1")
    blocks = build_system_blocks(_synthetic_schema())
    log = tmp_path / "llm_calls.jsonl"

    LLMClient(sdk_client=FakeAnthropic()).complete(
        blocks, "Q: hi", output_format=GeneratedSQL, log=log
    )

    second = FakeAnthropic()
    with pytest.raises(RequestCapExceeded):
        LLMClient(sdk_client=second).complete(
            blocks, "Q: hi", output_format=GeneratedSQL, log=log
        )
    assert second.calls == []


# ------------------------------------------------------------------ logging


def test_log_line_has_every_required_field_and_no_prompt_text(tmp_path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    fake = FakeAnthropic(usage=FakeUsage(inp=1200, out=340, write=2500, read=0))
    secret_question = "Q: a very distinctive question string"

    LLMClient(sdk_client=fake).complete(
        build_system_blocks(_synthetic_schema()),
        secret_question,
        output_format=GeneratedSQL,
        log=log,
    )

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])

    assert REQUIRED_LOG_FIELDS <= set(entry)
    assert entry["model"] == "claude-sonnet-5"
    assert entry["input_tokens"] == 1200
    assert entry["output_tokens"] == 340
    assert entry["cache_creation_input_tokens"] == 2500
    assert entry["cache_read_input_tokens"] == 0
    assert entry["estimated_cost_usd"] > 0
    assert entry["latency_ms"] >= 0
    assert len(entry["prompt_sha256"]) == 64

    # The hash exists so the prompt does not have to be stored.
    assert "distinctive question" not in lines[0]
    assert "widgets" not in lines[0]


def test_running_total_sums_the_log_and_survives_a_corrupt_tail(tmp_path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    log.write_text(
        '{"estimated_cost_usd": 0.01}\n'
        '{"estimated_cost_usd": 0.02}\n'
        "{ truncated mid-writ\n",
        encoding="utf-8",
    )
    assert running_total_usd(log) == pytest.approx(0.03)


def test_running_total_is_zero_when_no_log_exists(tmp_path) -> None:
    assert running_total_usd(tmp_path / "missing.jsonl") == 0.0


# ------------------------------------------------------------------- costing


def test_cost_prices_cache_writes_and_reads_at_their_own_rates() -> None:
    usage = FakeUsage(inp=1000, out=500, write=2000, read=4000)
    rate = PRICING["claude-sonnet-5"]

    expected = (
        1000 * rate.input_usd_per_mtok
        + 2000 * rate.input_usd_per_mtok * CACHE_WRITE_MULTIPLIER
        + 4000 * rate.input_usd_per_mtok * CACHE_READ_MULTIPLIER
        + 500 * rate.output_usd_per_mtok
    ) / 1_000_000

    assert estimate_cost_usd("claude-sonnet-5", usage) == pytest.approx(expected)
    # A cache read must be far cheaper than the same tokens fresh.
    assert estimate_cost_usd("claude-sonnet-5", FakeUsage(0, 0, 0, 10_000)) < estimate_cost_usd(
        "claude-sonnet-5", FakeUsage(10_000, 0, 0, 0)
    )


def test_unknown_model_raises_rather_than_costing_nothing() -> None:
    with pytest.raises(KeyError):
        estimate_cost_usd("claude-sonnet-5-20260101", FakeUsage())


def test_pricing_ids_carry_no_date_suffix() -> None:
    """A date-suffixed model id is a 404 at request time."""
    assert set(PRICING) == {"claude-sonnet-5", "claude-haiku-4-5"}
    assert client_mod.DEFAULT_MODEL == "claude-sonnet-5"


# ------------------------------------------------------------ GeneratedSQL


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", 1.5),
        ("confidence", -0.1),
        ("sql", "   "),
        ("sql", "DROP TABLE orders;"),
        ("sql", "UPDATE orders SET status = 'paid';"),
        ("sql", "-- harmless looking\nDELETE FROM orders;"),
    ],
)
def test_generated_sql_rejects_malformed_output(field, value) -> None:
    payload = {
        "sql": "SELECT 1;",
        "explanation": "e",
        "confidence": 0.5,
        "tables_used": [],
        "columns_used": [],
        "assumptions": [],
        "ambiguity": {"is_ambiguous": False, "interpretations": []},
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        GeneratedSQL(**payload)


def test_generated_sql_accepts_a_cte_and_a_commented_select() -> None:
    unambiguous = Ambiguity(is_ambiguous=False, interpretations=[])

    assert GeneratedSQL(
        sql="WITH t AS (SELECT 1 AS n) SELECT n FROM t;",
        explanation="e",
        confidence=1.0,
        tables_used=["t"],
        columns_used=["t.n"],
        assumptions=[],
        ambiguity=unambiguous,
    ).sql.startswith("WITH")

    assert "SELECT" in GeneratedSQL(
        sql="-- count them\n/* block */ SELECT count(*) FROM orders;",
        explanation="e",
        confidence=0.8,
        tables_used=["orders"],
        columns_used=[],
        assumptions=[],
        ambiguity=unambiguous,
    ).sql


# ------------------------------------------------------------------ wiring


def test_generate_sql_sends_the_expected_request_and_does_not_execute(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUERYGUARD_LLM_LOG", str(tmp_path / "llm_calls.jsonl"))
    fake = FakeAnthropic()

    answer, result = generate_sql_with_stats(
        "How many orders were cancelled?",
        client=LLMClient(sdk_client=fake),
        schema=_synthetic_schema(),
    )

    assert isinstance(answer, GeneratedSQL)
    assert result.cost_usd > 0

    (call,) = fake.calls
    assert call["model"] == "claude-sonnet-5"
    assert call["output_format"] is GeneratedSQL
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    # Sonnet 5 rejects these outright.
    assert "temperature" not in call and "budget_tokens" not in call
    # Only the user message varies; the cached prefix lives in `system`.
    assert call["messages"] == [
        {"role": "user", "content": "Q: How many orders were cancelled?"}
    ]
    assert len(call["system"]) == 3


# ---------------------------------------------------------------- ambiguity


def _ambiguity_payload(count: int) -> dict:
    return {
        "is_ambiguous": True,
        "interpretations": [
            {"label": f"r{i}", "sql": f"SELECT {i};", "explanation": "e"}
            for i in range(count)
        ],
    }


def test_an_ambiguous_response_becomes_a_clarification_not_a_query(
    tmp_path, monkeypatch
) -> None:
    """The whole point: no `.sql` is handed back for the caller to run."""
    monkeypatch.setenv("QUERYGUARD_LLM_LOG", str(tmp_path / "llm_calls.jsonl"))
    fake = FakeAnthropic(parsed=_ambiguous_answer())

    answer, result = generate_sql_with_stats(
        "What was our revenue last quarter?",
        client=LLMClient(sdk_client=fake),
        schema=_synthetic_schema(),
    )

    assert isinstance(answer, ClarificationNeeded)
    assert not isinstance(answer, GeneratedSQL)
    assert answer.question == "What was our revenue last quarter?"
    assert [option.label for option in answer.interpretations] == [
        "gross_revenue",
        "net_of_refunds",
    ]
    # Every reading must be runnable, not prose describing a query.
    assert all(option.sql.upper().startswith("SELECT") for option in answer.interpretations)
    # The call still happened and still cost money.
    assert result.cost_usd > 0


def test_an_unambiguous_response_still_returns_sql(tmp_path, monkeypatch) -> None:
    """The common path must not regress: one reading, one query, no clarification."""
    monkeypatch.setenv("QUERYGUARD_LLM_LOG", str(tmp_path / "llm_calls.jsonl"))
    fake = FakeAnthropic(parsed=_sample_answer())

    answer, _ = generate_sql_with_stats(
        "How many orders were cancelled?",
        client=LLMClient(sdk_client=fake),
        schema=_synthetic_schema(),
    )

    assert isinstance(answer, GeneratedSQL)
    assert answer.sql == "SELECT count(*) FROM orders WHERE status = 'cancelled';"
    assert answer.ambiguity.is_ambiguous is False
    assert answer.ambiguity.interpretations == []


@pytest.mark.parametrize("count", [0, 1, 4])
def test_ambiguous_needs_two_or_three_interpretations(count) -> None:
    with pytest.raises(ValidationError):
        Ambiguity(**_ambiguity_payload(count))


@pytest.mark.parametrize("count", [2, 3])
def test_two_or_three_interpretations_are_accepted(count) -> None:
    assert len(Ambiguity(**_ambiguity_payload(count)).interpretations) == count


def test_empty_top_level_sql_is_allowed_only_when_ambiguous() -> None:
    """`sql` is mandatory exactly when a single reading was actually chosen."""
    base = {
        "explanation": "e",
        "confidence": 0.0,
        "tables_used": [],
        "columns_used": [],
        "assumptions": [],
    }

    assert GeneratedSQL(sql="", ambiguity=_ambiguity_payload(2), **base).sql == ""

    with pytest.raises(ValidationError):
        GeneratedSQL(
            sql="", ambiguity={"is_ambiguous": False, "interpretations": []}, **base
        )


def test_an_interpretation_may_not_smuggle_in_a_write() -> None:
    """Interpretation SQL is held to the same standard as a lone answer."""
    for statement in ("DELETE FROM orders;", "-- innocent\nDROP TABLE orders;", "   "):
        with pytest.raises(ValidationError):
            Interpretation(label="l", sql=statement, explanation="e")


def test_surplus_interpretations_on_an_unambiguous_answer_are_ignored() -> None:
    """Tolerated on purpose: rejecting would discard usable SQL already paid for."""
    answer = Ambiguity(
        is_ambiguous=False,
        interpretations=[Interpretation(label="l", sql="SELECT 1;", explanation="e")],
    )
    assert answer.is_ambiguous is False


def test_clarification_renders_every_reading_with_its_query() -> None:
    result = CallResult(parsed=None, message=None, usage=FakeUsage(), cost_usd=0.01, latency_ms=5)
    answer = ClarificationNeeded(
        question="What was our revenue last quarter?",
        interpretations=_ambiguous_answer().ambiguity.interpretations,
    )

    rendered = _render_clarification(answer, result)

    assert "CLARIFICATION NEEDED - 2 defensible readings" in rendered
    for option in answer.interpretations:
        assert option.label in rendered
        assert option.explanation in rendered
        # The SQL is indented into the block, so match a distinctive line.
        assert option.sql.splitlines()[0].strip() in rendered
    assert "Cost        : $0.010000" in rendered


@pytest.mark.parametrize(
    "parsed,expected_code,expected_text",
    [
        (_sample_answer, 0, "SELECT count(*)"),
        (_ambiguous_answer, 3, "CLARIFICATION NEEDED"),
    ],
)
def test_cli_exit_code_distinguishes_a_clarification_from_a_query(
    parsed, expected_code, expected_text, tmp_path, monkeypatch, capsys
) -> None:
    """A caller must be able to branch on the outcome without parsing stdout."""
    monkeypatch.setenv("QUERYGUARD_LLM_LOG", str(tmp_path / "llm_calls.jsonl"))
    monkeypatch.setattr(
        generate_mod, "build_system_blocks", lambda schema=None: build_system_blocks(
            _synthetic_schema()
        )
    )
    monkeypatch.setattr(
        generate_mod, "LLMClient", lambda **kw: LLMClient(sdk_client=FakeAnthropic(parsed=parsed()))
    )

    code = generate_mod.main(["a question"])

    assert code == expected_code
    assert expected_text in capsys.readouterr().out
