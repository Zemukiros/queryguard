"""Tests for hallucination detection and confidence: back-translation, agreement,
and the v0 scorer.

No test here reaches the API -- conftest makes constructing a real SDK client
fail -- and none needs the database: agreement's executor is replaced where a
test gets that far. Result frames are shaped like real executor output:
Decimal for numeric, tz-aware datetime64 for timestamptz, datetime.date for
`::date`, int64 for counts.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from queryguard.executor import OUTCOME_FAILED, OUTCOME_OK, ExecutionResult
from queryguard.generate import Ambiguity, GeneratedSQL, Interpretation
from queryguard.llm.client import LLMClient, reset_request_count
from queryguard.schema.introspect import ColumnInfo, DatabaseSchema, TableInfo
from queryguard.validation.agreement import (
    AGREE,
    DISAGREE,
    INCOMPARABLE,
    check_agreement,
    compare_results,
    implies_ordering,
    is_non_trivial,
)
from queryguard.validation.backtranslate import (
    VALIDATION_MODEL,
    AlignmentJudgement,
    BackTranslation,
    back_translate,
    build_backtranslation_blocks,
    judge_alignment,
)
from queryguard.validation.confidence import (
    CALIBRATION_PATH,
    SCORER_VERSION,
    V0_VERSION,
    V0_WEIGHTS,
    WEIGHTS,
    Features,
    build_features,
    encode,
    load_weights,
    log_features,
    row_count_bucket,
    score,
)
from queryguard.validation.sanity import SanityFlag


@pytest.fixture(autouse=True)
def _reset_counter():
    reset_request_count()
    yield
    reset_request_count()


class _Usage:
    input_tokens = 500
    output_tokens = 40
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Message:
    def __init__(self, parsed) -> None:
        self.parsed_output = parsed
        self.usage = _Usage()


class _Messages:
    def __init__(self, parsed) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _Message(self.parsed)


class _Fake:
    def __init__(self, parsed) -> None:
        self.messages = _Messages(parsed)


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableInfo(
                name="customers",
                row_count=500,
                columns=[
                    ColumnInfo(name="customer_id", sql_type="integer", nullable=False, is_primary_key=True),
                    ColumnInfo(
                        name="lifetime_value",
                        sql_type="numeric(12,2)",
                        nullable=True,
                        comment="Stored running total across all years.",
                    ),
                ],
            )
        ],
        extracted_at=datetime.now(timezone.utc),
    )


def _answer(sql: str) -> GeneratedSQL:
    return GeneratedSQL(
        sql=sql,
        explanation="x",
        confidence=0.9,
        tables_used=[],
        columns_used=[],
        assumptions=[],
        ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]),
    )


def _ok(frame: pd.DataFrame, *, truncated: bool = False) -> ExecutionResult:
    return ExecutionResult(
        outcome=OUTCOME_OK, sql_sha256="0" * 64, rows=frame, row_count=len(frame), truncated=truncated
    )


# ------------------------------------------------------------ back-translation


WRONG_SQL = (
    "SELECT c.customer_id, c.lifetime_value FROM customers AS c "
    "ORDER BY c.lifetime_value DESC LIMIT 5"
)
QUESTION = "Which 5 customers spent the most in 2025?"


def test_back_translation_never_sees_the_original_question() -> None:
    """The blindness is the design: the question must appear nowhere in the request."""
    fake = _Fake(BackTranslation(question="Top 5 customers by lifetime value", details=[]))
    translated, call = back_translate(
        WRONG_SQL, client=LLMClient(sdk_client=fake, model=VALIDATION_MODEL), schema=_schema()
    )

    [request] = fake.messages.calls
    assert request["model"] == VALIDATION_MODEL
    assert request["output_format"] is BackTranslation
    assert WRONG_SQL in request["messages"][0]["content"]
    sent = json.dumps(request["system"]) + json.dumps(request["messages"])
    assert QUESTION not in sent and "2025" not in sent
    assert translated.question == "Top 5 customers by lifetime value"
    assert call.cost_usd > 0


def test_the_back_translation_prefix_is_the_schema_with_one_breakpoint_at_the_end() -> None:
    blocks = build_backtranslation_blocks(_schema())
    assert "lifetime_value" in blocks[0]["text"]
    assert "Stored running total" in blocks[0]["text"], "column comments are what expose a wrong column"
    assert [("cache_control" in b) for b in blocks] == [False, True]


def test_the_judge_sees_both_questions_and_no_schema() -> None:
    fake = _Fake(
        AlignmentJudgement(
            alignment=0.2,
            discrepancies=["original asks for 2025 spend; the query ranks by lifetime value"],
        )
    )
    translated = BackTranslation(
        question="Which 5 customers have the highest lifetime value?",
        details=["ranked by customers.lifetime_value", "no date filter"],
    )
    judgement, _ = judge_alignment(
        QUESTION, translated, client=LLMClient(sdk_client=fake, model=VALIDATION_MODEL)
    )

    [request] = fake.messages.calls
    content = request["messages"][0]["content"]
    assert QUESTION in content and translated.question in content
    assert "no date filter" in content
    assert "Database schema" not in json.dumps(request["system"])
    assert judgement.alignment == 0.2
    assert len(judgement.discrepancies) == 1


def test_the_judge_is_told_what_is_not_a_discrepancy() -> None:
    """Live run (e): a correct answer lost 1.5 logits to an extra column and a sort."""
    fake = _Fake(AlignmentJudgement(alignment=1.0, discrepancies=[]))
    judge_alignment(
        "What was the average order value by country?",
        BackTranslation(question="Average order value and order count by country, sorted descending", details=[]),
        client=LLMClient(sdk_client=fake, model=VALIDATION_MODEL),
    )
    system = " ".join(block["text"] for block in fake.messages.calls[0]["system"])
    flat = " ".join(system.split())
    assert "These are NOT discrepancies" in flat
    assert "extra columns the original did not ask for" in flat
    assert "column names or aliases" in flat
    assert "a sort order the original did not ask about" in flat
    assert "row cap" in flat


def test_alignment_is_bounded_zero_to_one() -> None:
    with pytest.raises(ValueError):
        AlignmentJudgement(alignment=1.5, discrepancies=[])


# ------------------------------------------------------------ which queries


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM orders WHERE status = 'cancelled'",
        "SELECT o.order_id FROM orders AS o JOIN customers AS c ON c.customer_id = o.customer_id",
        "SELECT o.order_id FROM orders AS o LEFT JOIN refunds AS r ON r.order_id = o.order_id",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "SELECT o.order_id FROM orders AS o WHERE o.customer_id IN (SELECT c.customer_id FROM customers AS c)",
        "SELECT o.status FROM orders AS o GROUP BY o.status",
        "SELECT o.order_id FROM orders AS o ORDER BY o.order_date DESC LIMIT 5",
        "SELECT c.customer_id FROM customers AS c ORDER BY c.lifetime_value DESC LIMIT 5",
    ],
)
def test_joins_aggregates_ctes_subqueries_and_top_n_are_non_trivial(sql) -> None:
    assert is_non_trivial(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT c.customer_id, c.email FROM customers AS c WHERE c.customer_id = 42",
        "SELECT round(p.price, 2) AS price FROM products AS p",
        # Sorted but not cut: every row comes back whatever the ranking.
        "SELECT o.order_id FROM orders AS o ORDER BY o.order_date DESC",
        # Cut but not ranked: an arbitrary sample, nothing a rewrite could check.
        "SELECT o.order_id FROM orders AS o LIMIT 5",
    ],
)
def test_a_plain_lookup_is_trivial(sql) -> None:
    assert not is_non_trivial(sql)


def test_ranking_words_imply_ordering() -> None:
    assert implies_ordering("Which 5 customers spent the most in 2025?")
    assert implies_ordering("List the latest orders")
    assert not implies_ordering("What was the average order value by country?")


# ------------------------------------------------------------ comparing results


def _spend(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["customer_id", "spent"]).astype({"customer_id": "int64"})


def test_the_same_rows_in_a_different_order_agree_when_order_does_not_matter() -> None:
    first = pd.DataFrame({"country": ["USA", "Japan", "Brazil"], "avg_value": [Decimal("3801.22"), Decimal("3644.10"), Decimal("3702.95")]})
    second = first.iloc[[2, 0, 1]].reset_index(drop=True)
    outcome, explanation = compare_results(first, second, ordered=False)
    assert outcome == AGREE
    assert "ignoring row order" in explanation


def test_order_matters_when_the_question_ranks() -> None:
    first = _spend((7, Decimal("9100.00")), (3, Decimal("8800.00")))
    second = _spend((3, Decimal("8800.00")), (7, Decimal("9100.00")))
    outcome, explanation = compare_results(first, second, ordered=True)
    assert outcome == DISAGREE
    assert "row 1" in explanation


def test_column_names_are_ignored() -> None:
    first = _spend((7, Decimal("9100.00")))
    second = first.rename(columns={"customer_id": "id", "spent": "total_2025"})
    assert compare_results(first, second, ordered=True)[0] == AGREE


def test_columns_in_a_different_order_are_matched_by_content() -> None:
    first = _spend((7, Decimal("9100.00")), (3, Decimal("8800.00")))
    second = first[["spent", "customer_id"]]
    outcome, explanation = compare_results(first, second, ordered=True)
    assert outcome == AGREE
    assert "matched by content" in explanation


def test_decimal_float_and_int_compare_within_relative_tolerance() -> None:
    first = pd.DataFrame({"n": pd.Series([1250], dtype="int64"), "total": [Decimal("1000000.00")]})
    second = pd.DataFrame({"n": [1250.0], "total": [1000000.0000004]})
    assert compare_results(first, second, ordered=False)[0] == AGREE


def test_a_difference_beyond_tolerance_disagrees_and_says_where() -> None:
    first = pd.DataFrame({"revenue": [Decimal("18887029.25")]})
    second = pd.DataFrame({"revenue": [Decimal("17402311.60")]})
    outcome, explanation = compare_results(first, second, ordered=False)
    assert outcome == DISAGREE
    assert "18,887,029.25" in explanation and "17,402,311.60" in explanation


def test_a_rounded_average_agrees_with_an_unrounded_one_and_says_so() -> None:
    first = pd.DataFrame({"country": ["USA"], "avg": [Decimal("38.22")]})
    second = pd.DataFrame({"country": ["USA"], "avg": [Decimal("38.2213765")]})
    outcome, explanation = compare_results(first, second, ordered=False)
    assert outcome == AGREE
    assert "after rounding" in explanation


def test_rounding_never_makes_a_count_equal_an_average() -> None:
    first = pd.DataFrame({"x": pd.Series([3], dtype="int64")})
    second = pd.DataFrame({"x": [Decimal("3.4")]})
    assert compare_results(first, second, ordered=False)[0] == DISAGREE


def test_timestamptz_and_date_compare_as_instants() -> None:
    first = pd.DataFrame({"month": pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True), "n": [10, 12]})
    second = pd.DataFrame({"month": [date(2025, 1, 1), date(2025, 2, 1)], "n": [Decimal("10"), Decimal("12")]})
    assert compare_results(first, second, ordered=True)[0] == AGREE


def test_nulls_equal_nulls() -> None:
    first = pd.DataFrame({"city": ["Lyon", None], "n": [3, 4]})
    second = pd.DataFrame({"city": [None, "Lyon"], "n": [4, 3]})
    assert compare_results(first, second, ordered=False)[0] == AGREE


def test_different_column_counts_are_incomparable() -> None:
    first = _spend((7, Decimal("9100.00")))
    second = first.assign(name="Mia Chen")
    assert compare_results(first, second, ordered=True)[0] == INCOMPARABLE


def test_different_row_counts_disagree() -> None:
    first = pd.DataFrame({"n": pd.Series([812], dtype="int64")})
    second = pd.DataFrame({"n": pd.Series([], dtype="int64")})
    outcome, explanation = compare_results(first, second, ordered=False)
    assert outcome == DISAGREE
    assert "1 vs 0" in explanation


# ------------------------------------------------------------ check_agreement


def _second(sql: str) -> LLMClient:
    return LLMClient(sdk_client=_Fake(_answer(sql)))


def test_check_agreement_runs_the_second_query_and_compares(monkeypatch) -> None:
    frames = {
        "first": pd.DataFrame({"n": pd.Series([812], dtype="int64")}),
        "second": pd.DataFrame({"count": pd.Series([812], dtype="int64")}),
    }
    monkeypatch.setattr("queryguard.validation.agreement.execute", lambda sql, cfg=None: _ok(frames["second"]))
    result = check_agreement(
        "How many orders were cancelled?",
        "SELECT count(*) AS n FROM orders AS o WHERE o.status = 'cancelled'",
        _ok(frames["first"]),
        client=_second("WITH c AS (SELECT o.order_id FROM orders AS o WHERE o.status = 'cancelled') SELECT count(*) FROM c"),
        schema=_schema(),
    )
    assert result.outcome == AGREE
    assert result.second_sql.startswith("WITH c AS")
    assert result.call is not None


def test_the_second_prompt_shows_the_first_query_as_contrast_only() -> None:
    fake = _Fake(_answer("SELECT 1"))
    from queryguard.validation.agreement import second_opinion

    second_opinion(QUESTION, WRONG_SQL, client=LLMClient(sdk_client=fake), schema=_schema())
    content = fake.messages.calls[0]["messages"][0]["content"]
    assert content.startswith(f"Q: {QUESTION}")
    assert WRONG_SQL in content
    assert "may be wrong" in content and "structurally different" in content


def test_an_ambiguous_second_answer_is_incomparable() -> None:
    ambiguous = GeneratedSQL(
        sql="",
        explanation="two readings",
        confidence=0.0,
        tables_used=[],
        columns_used=[],
        assumptions=[],
        ambiguity=Ambiguity(
            is_ambiguous=True,
            interpretations=[
                Interpretation(label="a", sql="SELECT 1", explanation="a"),
                Interpretation(label="b", sql="SELECT 2", explanation="b"),
            ],
        ),
    )
    result = check_agreement(
        QUESTION, WRONG_SQL, _ok(pd.DataFrame({"x": [1]})),
        client=LLMClient(sdk_client=_Fake(ambiguous)), schema=_schema(),
    )
    assert result.outcome == INCOMPARABLE


def test_a_blocked_second_query_is_incomparable_and_never_runs(monkeypatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr("queryguard.validation.agreement.execute", lambda *a, **k: ran.append("ran"))
    result = check_agreement(
        QUESTION, WRONG_SQL, _ok(pd.DataFrame({"x": [1]})),
        client=_second("WITH gone AS (DELETE FROM orders RETURNING *) SELECT count(*) FROM gone"),
        schema=_schema(),
    )
    assert result.outcome == INCOMPARABLE
    assert "blocked by forbidden_construct" in result.explanation
    assert ran == []


def test_a_failed_second_query_is_incomparable(monkeypatch) -> None:
    failed = ExecutionResult(outcome=OUTCOME_FAILED, sql_sha256="0" * 64, sqlstate="42703", error_message="no such column")
    monkeypatch.setattr("queryguard.validation.agreement.execute", lambda *a, **k: failed)
    result = check_agreement(
        QUESTION, WRONG_SQL, _ok(pd.DataFrame({"x": [1]})),
        client=_second("SELECT o.nope FROM orders AS o JOIN customers AS c ON true LIMIT 5"),
        schema=_schema(),
    )
    assert result.outcome == INCOMPARABLE
    assert "42703" in result.explanation


def test_a_truncated_result_is_incomparable(monkeypatch) -> None:
    frame = pd.DataFrame({"x": range(1000)})
    monkeypatch.setattr("queryguard.validation.agreement.execute", lambda *a, **k: _ok(frame, truncated=True))
    result = check_agreement(
        "List every order item", "SELECT * FROM order_items", _ok(frame, truncated=True),
        client=_second("SELECT oi.* FROM order_items AS oi JOIN orders AS o ON o.order_id = oi.order_id"),
        schema=_schema(),
    )
    assert result.outcome == INCOMPARABLE
    assert "truncated" in result.explanation


# ------------------------------------------------------------------ confidence


def _features(**overrides) -> Features:
    base = dict(
        executed=True, self_confidence=0.9, alignment=0.95, discrepancy_count=0,
        sanity_fail=0, sanity_warn=0, sanity_info=0, agreement="agree",
        guardrail_rewrote=False, row_count_bucket="1",
    )
    return Features(**{**base, **overrides})


def test_build_features_counts_flags_by_severity() -> None:
    flags = [
        SanityFlag("negative_values", "fail", "x"),
        SanityFlag("null_heavy_column", "warn", "x"),
        SanityFlag("constant_column", "warn", "x"),
        SanityFlag("null_heavy_column", "info", "x"),
    ]
    features = build_features(
        executed=True, self_confidence=0.8, alignment=0.5, discrepancies=["a", "b"],
        sanity=flags, agreement=None, guardrail_rewrote=True, row_count=1000, truncated=True,
    )
    assert (features.sanity_fail, features.sanity_warn, features.sanity_info) == (1, 2, 1)
    assert features.discrepancy_count == 2
    assert features.agreement == "not_run"
    assert features.row_count_bucket == "capped"


@pytest.mark.parametrize(
    "rows,truncated,bucket",
    [(0, False, "0"), (1, False, "1"), (5, False, "2-10"), (40, False, "11-100"), (1000, False, "101-999"), (1000, True, "capped")],
)
def test_row_count_buckets(rows, truncated, bucket) -> None:
    assert row_count_bucket(rows, truncated) == bucket


def test_the_encoded_vector_has_a_weight_for_every_feature() -> None:
    assert set(encode(_features())) == set(V0_WEIGHTS) - {"bias"}


def test_v0_a_clean_agreeing_run_scores_high() -> None:
    confidence, breakdown = score(_features(), V0_WEIGHTS)
    assert confidence > 0.9
    assert breakdown["agreement_agree"] == V0_WEIGHTS["agreement_agree"]


@pytest.mark.parametrize("weights", [V0_WEIGHTS, None], ids=["v0", "runtime"])
def test_low_alignment_and_disagreement_score_low(weights) -> None:
    hallucinated = _features(alignment=0.2, discrepancy_count=2, agreement="disagree")
    confidence, breakdown = score(hallucinated, weights)
    assert confidence < 0.1
    assert breakdown["agreement_disagree"] < 0 and breakdown["alignment_centered"] < 0


BAD_SIGNALS = [
    {"alignment": 0.3}, {"discrepancy_count": 1}, {"sanity_fail": 1}, {"sanity_warn": 1},
    {"agreement": "disagree"}, {"agreement": "incomparable"}, {"row_count_bucket": "0"},
    {"alignment": None},
]


def test_v0_each_bad_signal_alone_lowers_the_score() -> None:
    clean, _ = score(_features(), V0_WEIGHTS)
    for override in [*BAD_SIGNALS, {"self_confidence": 0.4}]:
        assert score(_features(**override), V0_WEIGHTS)[0] < clean, override


def test_runtime_weights_each_bad_signal_alone_lowers_the_score() -> None:
    """Whatever the calibration fitted, no warning sign may raise the score."""
    clean, _ = score(_features())
    for override in BAD_SIGNALS:
        assert score(_features(**override))[0] < clean, override


def test_the_committed_calibration_is_what_runs() -> None:
    assert CALIBRATION_PATH.exists()
    assert SCORER_VERSION.startswith("calibrated-")
    assert WEIGHTS == load_weights()[0] and set(WEIGHTS) == set(V0_WEIGHTS)
    # Fitted without self-confidence: it is an injected constant on most training rows.
    assert WEIGHTS["self_confidence"] == 0.0


def test_a_missing_calibration_falls_back_to_v0(tmp_path) -> None:
    assert load_weights(tmp_path / "absent.json") == (V0_WEIGHTS, V0_VERSION)


def test_a_calibration_for_other_features_is_refused(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": "x", "intercept": 0.0, "weights": {"self_confidence": 1.0}}))
    with pytest.raises(ValueError, match="do not match"):
        load_weights(path)


def test_an_unexecuted_query_scores_zero() -> None:
    assert score(_features(executed=False)) == (0.0, {})


def test_every_scored_run_is_logged_as_a_training_row(tmp_path) -> None:
    path = tmp_path / "features.jsonl"
    features = _features(agreement="disagree")
    confidence, breakdown = score(features)
    log_features(question=QUESTION, sql=WRONG_SQL, features=features, confidence=confidence, breakdown=breakdown, path=path)

    [row] = [json.loads(line) for line in path.read_text().splitlines()]
    assert row["scorer_version"] == SCORER_VERSION
    assert row["question"] == QUESTION and row["sql"] == WRONG_SQL
    assert row["features"]["agreement"] == "disagree"
    assert row["encoded"]["agreement_disagree"] == 1.0
    assert row["confidence"] == pytest.approx(confidence)
    assert row["label"] is None, "left for a human or eval suite to fill in"


# ------------------------------------------- extra columns (first eval run fix)


def test_either_side_may_add_columns_when_the_rest_matches() -> None:
    narrow = pd.DataFrame({"category": ["Apparel", "Books"], "n": [31, 18]})
    wide = pd.DataFrame({"id": [3, 5], "name": ["Apparel", "Books"], "count": [31, 18]})
    for first, second in ((narrow, wide), (wide, narrow)):
        outcome, explanation = compare_results(first, second, ordered=False, extra_columns="either")
        assert outcome == AGREE
        assert "1 extra column(s) ignored" in explanation


def test_by_default_a_different_width_is_still_incomparable() -> None:
    narrow = pd.DataFrame({"n": [1]})
    assert compare_results(narrow, narrow.assign(x=2), ordered=False)[0] == INCOMPARABLE


def test_second_may_add_columns_but_never_drop_them() -> None:
    golden = pd.DataFrame({"id": [1, 2], "total": [Decimal("5.00"), Decimal("7.00")]})
    generated = golden.assign(name=["a", "b"])
    assert compare_results(golden, generated, ordered=False, extra_columns="second")[0] == AGREE
    assert compare_results(generated, golden, ordered=False, extra_columns="second")[0] == INCOMPARABLE


def test_extra_columns_do_not_hide_a_wrong_value() -> None:
    golden = pd.DataFrame({"id": [1, 2], "total": [Decimal("5.00"), Decimal("7.00")]})
    generated = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "total": [Decimal("5.00"), Decimal("9.99")]})
    outcome, explanation = compare_results(golden, generated, ordered=False, extra_columns="second")
    assert outcome == INCOMPARABLE
    assert "do not all appear" in explanation


def test_a_row_count_difference_with_extra_columns_is_disagreement() -> None:
    first = pd.DataFrame({"n": [1, 2]})
    second = pd.DataFrame({"n": [1], "label": ["x"]})
    assert compare_results(first, second, ordered=False, extra_columns="either")[0] == DISAGREE


def test_last_name_and_first_quarter_do_not_imply_ordering() -> None:
    """'last name' made a 14-row lookup compare in order in the first eval run."""
    assert not implies_ordering(
        "List every order placed by customer 42 with its date, total and the customer's last name."
    )
    assert not implies_ordering("How many orders in the first quarter of 2026?")
    assert not implies_ordering("Count orders by country")
    assert implies_ordering("Which 5 customers spent the most?")
    assert implies_ordering("List products ordered by price")


def test_a_logged_second_query_can_be_re_evaluated_without_the_api(monkeypatch) -> None:
    from queryguard.validation.agreement import evaluate_second_sql

    second = pd.DataFrame({"name": ["Mia"], "n": pd.Series([812], dtype="int64")})
    monkeypatch.setattr("queryguard.validation.agreement.execute", lambda *a, **k: _ok(second))
    result = evaluate_second_sql(
        "How many orders were cancelled?",
        "SELECT 'Mia' AS name, count(*) AS n FROM orders AS o JOIN customers AS c ON true",
        _ok(pd.DataFrame({"cancelled": pd.Series([812], dtype="int64")})),
    )
    assert result.outcome == AGREE
    assert result.call is None
