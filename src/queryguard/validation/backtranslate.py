"""Blind back-translation: what question does this SQL actually answer?

A generated query that runs cleanly can still answer a different question from
the one asked -- gross revenue for net, a customer's lifetime value for their
spend in one year. The sanity checks cannot see that, because the rows look
fine. Reading the SQL back into English and comparing the two questions can.

Two calls, both on Haiku:

1. Back-translation sees the SQL and the schema, and NOT the original question.
   That blindness is the design. A model shown "Which customers spent the most
   in 2025?" next to a query over `lifetime_value` reads the query charitably
   and reports that it answers the question -- the bias this step exists to
   avoid. Shown only the SQL, it has to say what the SQL does.
2. The judge sees the two questions and no schema. Its job is semantic
   comparison of two English sentences; the schema would only invite it to
   re-derive the SQL, which is the back-translator's job.

The back-translation prefix is the schema plus fixed instructions, with the
breakpoint on the last block, and nothing else: no few-shot examples, because
an example question is a question the model could echo when the SQL resembles
its query. Haiku 4.5's minimum cacheable prefix is 4096 tokens and the schema is
close to that, so whether this prefix caches is measured, not assumed -- a
prefix below the minimum is silently not cached, and the client's
`_warn_if_cache_was_ignored` is what reports it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from queryguard.llm.client import CallResult, LLMClient
from queryguard.schema.introspect import DatabaseSchema

VALIDATION_MODEL = "claude-haiku-4-5"

BACKTRANSLATE_INSTRUCTIONS = """\
You read a PostgreSQL query and state, in one plain-English question, exactly
what it answers. You are not told what question the query was written for, and
you must not guess at one: describe what the SQL does, not what it was probably
meant to do.

Be precise about everything that changes the answer:
- the metric and how it is computed (count of rows vs. count of distinct
  values, sum of which column, gross vs. net of refunds);
- every filter, including status filters, and what they include or exclude;
- the time window, and which date column defines it;
- grouping, ordering and any row limit that is part of the question (a top-N);
- which column a value comes from, using the schema comments to say what that
  column really holds -- a stored running total is not a sum over orders.

Write the question the way a business user would ask it, but never drop a
detail to make it read better. Use `details` to list each filter, definition
and window as a short separate item."""

JUDGE_INSTRUCTIONS = """\
You compare two questions about the same e-commerce database: the ORIGINAL
question a user asked, and a BACK-TRANSLATED question describing what a SQL
query actually computes. Decide whether answering the second answers the first.

Score `alignment` from 0 to 1:
- 1.0: same meaning. Wording, extra columns, column names, and sorting the
  original did not ask about may all differ.
- 0.7-0.9: same question with a minor difference that rarely changes the answer.
- 0.4-0.6: one material difference -- a different time window, a missing or
  extra filter, a different metric definition.
- 0.0-0.3: a different question, or several material differences.

List every material difference in `discrepancies`, one per item, each saying
what the original asks for and what the query does instead, e.g. "original
asks for net revenue; the query computes gross revenue". Leave it empty when
the questions match. Do not invent differences: a reasonable reading of an
underspecified original (e.g. treating "orders" as all orders) is not one.

These are NOT discrepancies -- do not list them and do not lower the score:
- extra columns the original did not ask for (a count beside an average, a
  name beside an id);
- column names or aliases;
- a sort order the original did not ask about;
- a row cap such as LIMIT 1000 or 1001 added for safety."""


class BackTranslation(BaseModel):
    """What a query answers, stated without knowledge of why it was written."""

    question: str = Field(description="The single question this SQL answers, in plain English.")
    details: list[str] = Field(
        description="Each filter, metric definition, time window and limit, one per item."
    )


class AlignmentJudgement(BaseModel):
    """How well the query's question matches the one that was asked."""

    alignment: float = Field(ge=0.0, le=1.0, description="0-1; 1 means the same question.")
    discrepancies: list[str] = Field(
        description="Each material difference: what the original asks vs. what the query does."
    )


def build_backtranslation_blocks(schema: DatabaseSchema) -> list[dict[str, Any]]:
    """Schema then instructions; one breakpoint, on the last block."""
    # Below Haiku 4.5's 4,096-token cache floor by design ("SQL plus schema only"), so this never caches.
    return [
        {"type": "text", "text": "Database schema:\n\n" + schema.render_for_prompt()},
        {
            "type": "text",
            "text": BACKTRANSLATE_INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def back_translate(
    sql: str, *, client: LLMClient, schema: DatabaseSchema
) -> tuple[BackTranslation, CallResult]:
    """One Haiku call. The original question is deliberately not a parameter."""
    result = client.complete(
        build_backtranslation_blocks(schema),
        f"SQL:\n{sql}",
        output_format=BackTranslation,
        max_tokens=1024,
    )
    return result.parsed, result


def judge_alignment(
    original: str, back_translated: BackTranslation, *, client: LLMClient
) -> tuple[AlignmentJudgement, CallResult]:
    """One Haiku call comparing two questions. No schema: see the module docstring."""
    details = "\n".join(f"- {d}" for d in back_translated.details) or "- (none)"
    message = (
        f"ORIGINAL: {original}\n\n"
        f"BACK-TRANSLATED: {back_translated.question}\n"
        f"Details of what the query computes:\n{details}"
    )
    result = client.complete(
        [{"type": "text", "text": JUDGE_INSTRUCTIONS}],
        message,
        output_format=AlignmentJudgement,
        max_tokens=1024,
    )
    return result.parsed, result
