"""Known-wrong variants of every golden query: labelled negatives for calibration.

    uv run python -m evals.mutations

Each mutator makes one systematic, plausible mistake -- the kind a model
actually makes -- to a golden query:

  literal_case   'cancelled' -> 'Cancelled'            (case-sensitive match)
  drop_where     drop one WHERE condition              (a forgotten filter)
  agg_swap       count(DISTINCT x) -> count(x), sum -> avg, avg -> sum,
                 count(*) -> count(grouped column)     (NULL group counts 0)
  column_swap    a plausible neighbour column          (total_amount -> lifetime_value)
  inner_to_left  JOIN -> LEFT JOIN                      (outer join keeps non-matches)
  date_shift     a date bound moved back one year      (off-by-a-year window)
  order_flip     ORDER BY ... DESC <-> ASC             (top-N from the wrong end)
  fan_out_join   join a child table before aggregating (each parent counted N times)
  null_flip      IS NULL <-> IS NOT NULL, NOT EXISTS <-> EXISTS   (anti-join inverted)

A mutation whose result runs past the row cap is kept: the golden answer fits
under the cap, so a truncated one is a different answer by construction.

A mutation is kept only if it changes the answer. The comparison is
`compare_results` from the agreement detector, so "different" means exactly
what it means to the pipeline: beyond 1e-6 relative tolerance, row order
ignored unless the golden entry is `ordered`, column names ignored. A mutation
that returns the golden answer is not wrong, whatever its SQL looks like --
LEFT JOIN on a NOT NULL foreign key is the common case -- and keeping it would
label a correct answer as a hallucination. A mutation the guardrail blocks or
the database rejects is discarded too: an error is a different failure from a
wrong answer, and these are negatives for the wrong-answer detector.

Mutators are tried in an order rotated per question, so no type is starved by
the cap of three kept per question.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable

import sqlparse
from sqlparse.sql import TokenList, Where
from sqlparse import tokens as T

from evals.common import GOLDEN_RESULTS, MUTATION_RESULTS, load_golden, result_hash, run_guarded
from queryguard.validation.agreement import AGREE, compare_results

MAX_KEPT_PER_QUESTION = 3

# Matches whole literals, DATE ones included, so scanning never starts inside
# one: a lookbehind for DATE skipped the opening quote but then matched from
# the date's closing quote to the next literal's opening quote.
_STRING = re.compile(r"(DATE\s+)?'((?:[^']|'')*)'", re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_LITERAL = re.compile(r"DATE '(\d{4})-(\d{2}-\d{2})'", re.IGNORECASE)
_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-z_]\w*)(?:\s+(?:AS\s+)?([a-z_]\w*))?", re.IGNORECASE
)
_NOT_AN_ALIAS = {"where", "join", "on", "left", "right", "inner", "outer", "full", "cross",
                 "group", "order", "limit", "having", "using"}

# (table, column) -> (table, column): the plausible wrong column a model reaches for.
NEIGHBOURS: dict[tuple[str, str], tuple[str, str]] = {
    ("orders", "total_amount"): ("customers", "lifetime_value"),
    ("orders", "order_date"): ("orders", "shipped_at"),
    ("orders", "shipped_at"): ("orders", "order_date"),
    ("order_items", "unit_price"): ("products", "price"),
    ("order_items", "quantity"): ("products", "stock_quantity"),
    ("products", "price"): ("products", "cost"),
    ("refunds", "amount"): ("orders", "total_amount"),
    ("refunds", "refunded_at"): ("orders", "order_date"),
    ("customers", "country"): ("customers", "city"),
    ("customers", "city"): ("customers", "country"),
    ("customers", "email"): ("customers", "phone"),
    ("customers", "signup_date"): ("orders", "order_date"),
    ("customers", "marketing_opt_in"): ("customers", "is_active"),
    ("refunds", "order_id"): ("refunds", "order_item_id"),
}


# ------------------------------------------------------------------ mutators


def literal_case(sql: str) -> str | None:
    """Flip the case of the first string literal that has letters and is not a date."""
    for match in _STRING.finditer(sql):
        value = match.group(2)
        if match.group(1) or _ISO_DATE.match(value) or not re.search(r"[A-Za-z]", value):
            continue
        flipped = value.lower() if value != value.lower() else value.capitalize()
        if flipped == value:
            flipped = value.upper()
        return sql[: match.start(2)] + flipped + sql[match.end(2):]
    return None


def _first_where(node) -> Where | None:
    for tok in getattr(node, "tokens", []):
        if isinstance(tok, Where):
            return tok
        if isinstance(tok, TokenList):
            found = _first_where(tok)
            if found is not None:
                return found
    return None


def drop_where(sql: str) -> str | None:
    """Drop one condition from the first WHERE: a non-date one if there is one."""
    statement = sqlparse.parse(sql)[0]
    where = _first_where(statement)
    if where is None:
        return None

    conjuncts: list[str] = []
    current: list[str] = []
    for tok in where.tokens[1:]:  # skip the WHERE keyword itself
        if tok.ttype is T.Keyword and tok.normalized == "AND":
            conjuncts.append("".join(current).strip())
            current = []
        else:
            current.append(str(tok))
    conjuncts.append("".join(current).strip())
    conjuncts = [c for c in conjuncts if c]

    original = str(where)
    trailing = original[len(original.rstrip()):]
    if len(conjuncts) == 1:
        replacement = trailing
    else:
        victim = next((c for c in conjuncts if "DATE '" not in c.upper()), conjuncts[0])
        kept = [c for c in conjuncts if c is not victim]
        replacement = "WHERE " + "\n  AND ".join(kept) + trailing
    return sql.replace(original, replacement, 1)


def agg_swap(sql: str) -> str | None:
    """The first applicable of: count(DISTINCT x) -> count(x), sum -> avg,
    avg -> sum, count(*) -> count(<first GROUP BY column>)."""
    for pattern, replacement in [
        (r"\bcount\(\s*DISTINCT\s+", "count("),
        (r"\bsum\(", "avg("),
        (r"\bavg\(", "sum("),
    ]:
        swapped = re.sub(pattern, replacement, sql, count=1, flags=re.IGNORECASE)
        if swapped != sql:
            return swapped
    group = re.search(r"\bGROUP\s+BY\s+([a-z_]\w*\.[a-z_]\w*)", sql, re.IGNORECASE)
    if group and re.search(r"\bcount\(\*\)", sql, re.IGNORECASE):
        # count(col) skips NULLs, so a NULL group (auto-approved refunds) counts 0.
        return re.sub(r"\bcount\(\*\)", f"count({group.group(1)})", sql, count=1, flags=re.IGNORECASE)
    return None


def _aliases(sql: str) -> dict[str, str]:
    """alias -> table, and table -> table, for every table in FROM / JOIN."""
    found: dict[str, str] = {}
    for match in _TABLE_REF.finditer(sql):
        table = match.group(1).lower()
        found.setdefault(table, table)
        alias = match.group(2)
        if alias and alias.lower() not in _NOT_AN_ALIAS:
            found[alias.lower()] = table
    return found


def column_swap(sql: str) -> str | None:
    """Replace every `alias.col` with a plausible neighbour whose table is in scope."""
    aliases = _aliases(sql)
    alias_for = {}
    for alias, table in aliases.items():
        if alias != table or table not in alias_for:
            alias_for[table] = alias
    for match in re.finditer(r"\b([a-z_]\w*)\.([a-z_]\w*)\b", sql):
        alias, column = match.group(1).lower(), match.group(2).lower()
        table = aliases.get(alias)
        target = NEIGHBOURS.get((table, column)) if table else None
        if target is None or target[0] not in alias_for:
            continue
        target_alias = alias if target[0] == table else alias_for[target[0]]
        return re.sub(
            rf"\b{re.escape(match.group(1))}\.{re.escape(match.group(2))}\b",
            f"{target_alias}.{target[1]}",
            sql,
        )
    return None


def inner_to_left(sql: str) -> str | None:
    """The first inner JOIN (bare or INNER JOIN) becomes LEFT JOIN."""
    for match in re.finditer(r"\b(?:INNER\s+)?JOIN\b", sql, re.IGNORECASE):
        preceding = sql[: match.start()].rstrip().split()
        if preceding and preceding[-1].upper() in {"LEFT", "RIGHT", "FULL", "CROSS", "OUTER"}:
            continue
        return sql[: match.start()] + "LEFT JOIN" + sql[match.end():]
    return None


def date_shift(sql: str) -> str | None:
    """Move the first DATE literal back one year."""
    match = _DATE_LITERAL.search(sql)
    if match is None:
        return None
    shifted = f"DATE '{int(match.group(1)) - 1}-{match.group(2)}'"
    return sql[: match.start()] + shifted + sql[match.end():]


def order_flip(sql: str) -> str | None:
    """Flip the direction of the outermost ORDER BY's first key."""
    matches = list(re.finditer(r"\bORDER\s+BY\b", sql, re.IGNORECASE))
    if not matches:
        return None
    start = matches[-1].end()
    tail = sql[start:]
    direction = re.search(r"\b(DESC|ASC)\b", tail, re.IGNORECASE)
    first_key_end = re.search(r",|\bLIMIT\b|$", tail, re.IGNORECASE).start()
    if direction and direction.start() < first_key_end:
        flipped = "ASC" if direction.group(1).upper() == "DESC" else "DESC"
        return sql[: start + direction.start()] + flipped + sql[start + direction.end():]
    # No explicit direction means ASC; make it DESC.
    head, tail = sql[: start + first_key_end], sql[start + first_key_end:]
    gap = head[len(head.rstrip()):]
    return f"{head.rstrip()} DESC{gap}{tail}"


# parent table -> (child table, join column): joining the child before an
# aggregate repeats each parent row once per child row.
FAN_OUT = {
    "orders": ("order_items", "order_id"),
    "customers": ("orders", "customer_id"),
    "products": ("order_items", "product_id"),
    "categories": ("products", "category_id"),
    "refunds": ("order_items", "order_id"),  # "join the line items to see what was refunded"
}


def fan_out_join(sql: str) -> str | None:
    """Join a child table straight after the first FROM whose table has children."""
    for match in _TABLE_REF.finditer(sql):
        if not match.group(0).upper().startswith("FROM"):
            continue
        table = match.group(1).lower()
        if table not in FAN_OUT:
            continue
        child, column = FAN_OUT[table]
        if re.search(rf"\b{child}\b", sql, re.IGNORECASE):
            continue  # already joined: the fan-out is part of the golden query
        alias = match.group(2) if match.group(2) and match.group(2).lower() not in _NOT_AN_ALIAS else table
        end = match.end() if alias != table or match.group(2) else match.end(1)
        join = f"\nJOIN {child} AS fan_{child} ON fan_{child}.{column} = {alias}.{column}"
        return sql[:end] + join + sql[end:]
    return None


def null_flip(sql: str) -> str | None:
    """Invert the first NULL test or (NOT) EXISTS."""
    match = re.search(r"\bIS\s+(NOT\s+)?NULL\b|\b(NOT\s+)?EXISTS\b", sql, re.IGNORECASE)
    if match is None:
        return None
    text = match.group(0).upper()
    if text.startswith("IS"):
        flipped = "IS NULL" if "NOT" in text else "IS NOT NULL"
    else:
        flipped = "EXISTS" if text.startswith("NOT") else "NOT EXISTS"
    return sql[: match.start()] + flipped + sql[match.end():]


MUTATORS: dict[str, Callable[[str], str | None]] = {
    "literal_case": literal_case,
    "drop_where": drop_where,
    "agg_swap": agg_swap,
    "column_swap": column_swap,
    "inner_to_left": inner_to_left,
    "date_shift": date_shift,
    "order_flip": order_flip,
    "fan_out_join": fan_out_join,
    "null_flip": null_flip,
}


# --------------------------------------------------------------------- driver


def mutate_entry(index: int, entry: dict, golden_frame, stats: dict) -> tuple[list[dict], list[dict]]:
    names = list(MUTATORS)
    rotation = names[index % len(names):] + names[: index % len(names)]
    ordered = bool(entry.get("ordered", False))
    kept: list[dict] = []
    discarded: list[dict] = []

    for name in rotation:
        if len(kept) >= MAX_KEPT_PER_QUESTION:
            break
        mutated = MUTATORS[name](entry["golden_sql"])
        if mutated is None or mutated.strip() == entry["golden_sql"].strip():
            stats[name]["not_applicable"] += 1
            continue
        stats[name]["attempted"] += 1
        run = run_guarded(mutated)
        if not run.ok:
            stats[name]["discarded_error"] += 1
            discarded.append({"golden_id": entry["id"], "mutation": name, "reason": run.error, "sql": mutated})
            continue
        frame = run.execution.rows
        outcome, explanation = compare_results(golden_frame, frame, ordered=ordered)
        if outcome == AGREE:
            stats[name]["discarded_same"] += 1
            discarded.append({"golden_id": entry["id"], "mutation": name, "reason": "same result as golden", "sql": mutated})
            continue
        stats[name]["kept"] += 1
        kept.append(
            {
                "id": f"{entry['id']}__{name}",
                "golden_id": entry["id"],
                "category": entry["category"],
                "question": entry["question"],
                "mutation": name,
                "sql": mutated,
                "label": "wrong",
                "row_count": len(frame),
                "result_sha256": result_hash(frame, ordered=ordered),
                "diff": f"{outcome}: {explanation}",
            }
        )
    return kept, discarded


def main() -> int:
    golden = json.loads(GOLDEN_RESULTS.read_text(encoding="utf-8"))["results"]
    stats: dict[str, Counter] = defaultdict(Counter)
    kept: list[dict] = []
    discarded: list[dict] = []
    per_question: dict[str, int] = {}

    answerable = [e for e in load_golden() if "golden_sql" in e]
    for index, entry in enumerate(answerable):
        run = run_guarded(entry["golden_sql"])
        if not run.ok:
            print(f"{entry['id']}: golden query failed ({run.error}); run evals.run_golden first")
            return 1
        ordered = bool(entry.get("ordered", False))
        if result_hash(run.execution.rows, ordered=ordered) != golden[entry["id"]]["result_sha256"]:
            print(f"{entry['id']}: golden result no longer matches golden_results.json; re-run evals.run_golden")
            return 1
        k, d = mutate_entry(index, entry, run.execution.rows, stats)
        kept += k
        discarded += d
        per_question[entry["id"]] = len(k)

    MUTATION_RESULTS.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "kept": kept,
                "discarded": discarded,
                "summary": {name: dict(stats[name]) for name in MUTATORS},
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{'mutation':<15}{'attempted':>10}{'kept':>6}{'same':>6}{'error':>7}{'n/a':>6}")
    for name in MUTATORS:
        s = stats[name]
        print(f"{name:<15}{s['attempted']:>10}{s['kept']:>6}{s['discarded_same']:>6}{s['discarded_error']:>7}{s['not_applicable']:>6}")
    counts = Counter(per_question.values())
    print(f"\n{len(kept)} kept negatives across {len(answerable)} golden queries; "
          f"kept per question: " + ", ".join(f"{n}->{counts[n]}" for n in sorted(counts)))
    short = [q for q, n in per_question.items() if n < 2]
    if short:
        print("fewer than 2 kept: " + ", ".join(f"{q} ({per_question[q]})" for q in short))
    return 0


if __name__ == "__main__":
    sys.exit(main())
