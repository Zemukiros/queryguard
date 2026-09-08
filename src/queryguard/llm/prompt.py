"""Assembles the system prompt.

Three blocks: fixed instructions, then the rendered schema, then the worked
examples. That is not stable-to-volatile order. The instructions and the
examples are module constants; the schema is introspected and changes whenever
the database does, so the volatile block is the middle one, not the last.

That costs nothing as things stand. All three blocks are invariant across
questions -- only the user message changes -- and the single breakpoint on the
last block caches the whole prefix, so re-introspecting the schema invalidates
that entire prefix whichever slot the schema occupies. The ordering would start
to matter only if a second breakpoint were added: the schema would then have to
move last, so that a schema change could not invalidate the blocks ahead of it.

Caching is a prefix match, so the breakpoint goes on block 3 and nowhere else.
Marking all three would consume three of the four available breakpoints and
cache exactly the same bytes.
"""

from __future__ import annotations

from typing import Any

from queryguard.llm.examples import render_examples
from queryguard.schema.introspect import DatabaseSchema, load_schema

SYSTEM_INSTRUCTIONS = """\
You translate natural-language questions into a single PostgreSQL 16 query.

Rules:
- Emit exactly one statement: a SELECT, or a WITH ... SELECT. Never INSERT,
  UPDATE, DELETE, TRUNCATE, CREATE, DROP, ALTER, GRANT or a transaction command.
  The database role that runs your SQL has SELECT and nothing else, so a write
  is not a risk to the data -- it is simply an error that returns no answer.
- Use PostgreSQL syntax and functions only.
- Write explicit JOIN ... ON. Never comma-join in the FROM clause.
- Alias every table and qualify every column with its alias.
- Only reference tables and columns that appear in the schema below. If the
  question needs something that is not there, say so in `explanation`, lower
  `confidence`, and write the closest query you honestly can. Never invent a
  column name that looks plausible.
- The column comments in the schema are authoritative about what a column
  holds. Read them before assuming what a business term means -- several
  columns behave differently than their names suggest.
- For date ranges use a half-open interval (>= start AND < day-after-end).
  BETWEEN on a timestamp silently drops rows with a time of day on the last day.
- Record every judgement call you made in `assumptions` -- an undefined period
  like "last quarter", a definition of revenue you had to choose, a status
  filter you decided to apply. If you assumed nothing, return an empty list.
- `confidence` is your own honest estimate that this query answers the question
  as asked: 1.0 only when the question is unambiguous and fully covered by the
  schema; below 0.5 when you had to guess at intent."""


def build_system_blocks(schema: DatabaseSchema | None = None) -> list[dict[str, Any]]:
    """The three system blocks, with cache_control on the last one only."""
    resolved = schema if schema is not None else load_schema()

    return [
        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
        {
            "type": "text",
            "text": "Database schema:\n\n" + resolved.render_for_prompt(),
        },
        {
            "type": "text",
            "text": render_examples(),
            "cache_control": {"type": "ephemeral"},
        },
    ]


def build_user_message(question: str) -> str:
    return f"Q: {question}"
