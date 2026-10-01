"""Static gate over SQL that is about to be executed.

The database is the real boundary: `queryguard_ro` holds SELECT and nothing
else, and `03_readonly_user.sql` calls `default_transaction_read_only` on top of
that "belt and braces". This module is the second boundary, in-process and
ahead of the connection, and it exists because the first one has two gaps a
read-only role cannot close:

1. `default_transaction_read_only` is a USERSET GUC, as `verify_db.py` says in
   as many words -- any role can flip it. Privileges still hold, but a guard
   that depends on that GUC is not a guard.
2. Privileges say nothing about cost. `SELECT * FROM orders` is entirely legal
   for a read-only role and will still happily stream every row into this
   process. A row cap is the only thing that prevents that, and it has to be
   imposed before execution, not after.

Every decision here is made on sqlparse's token stream, never on a substring of
the raw text. That distinction is the whole design: a column named `updated_at`
and a literal `'DROP TABLE users'` are both perfectly ordinary SQL, and a
blocklist grepping for `UPDATE` or `DROP` rejects them both. Tokens know the
difference between a keyword, an identifier and a string.

Two findings about sqlparse 0.6.0 shape the rules, both verified rather than
assumed:

- `get_type()` reports SELECT for a data-modifying CTE. `WITH x AS (INSERT INTO
  t VALUES (1) RETURNING *) SELECT * FROM x` is a write that answers "SELECT",
  so the statement type alone is not enough and DML is scanned tree-wide.
- Parenthesis depth is not subquery depth. `round(avg(x), 2)` nests two parens
  and zero subqueries; few-shot example 5 does exactly that. Only a paren that
  directly contains a SELECT counts as a level of nesting, or the depth limit
  rejects ordinary aggregate SQL.

Known limitation: this is not a PostgreSQL parser and does not try to be.
sqlparse accepts `SELECT * FROM (SELECT 1` without complaint; the balanced-paren
check catches that particular shape, but malformed SQL that slips past lands on
the server, where the read-only role makes the failure harmless. What must never
slip past is a *write*, and that is what the rules below are built around.

Nothing in this module touches a database or a network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Parenthesis, Statement, TokenList

# A thousand rows is far more than a person reads and small enough that a
# pathological query cannot exhaust memory before the cap bites.
DEFAULT_MAX_ROWS = 1000

# Three levels covers every shape the few-shot examples teach (the deepest, a
# CTE, reaches one) with room to spare. Deeper than that is either generated
# nonsense or a query nobody can review.
DEFAULT_MAX_SUBQUERY_DEPTH = 3

# Functions that read the filesystem, reach another host, sleep, or move large
# objects. None of these need a write privilege to do damage.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_logdir_ls",
        "copy",
        "query_to_xml",
    }
)

# Prefix matches, because these come in families: lo_import, lo_export, lo_get,
# dblink_exec, dblink_send_query.
FORBIDDEN_PREFIXES = ("lo_", "dblink")

# Keywords that have no business inside a SELECT. The statement-type rule already
# rejects these in leading position; this catches them anywhere, which is where a
# smuggled one would be.
FORBIDDEN_KEYWORDS = frozenset(
    {
        "ANALYSE",
        "ANALYZE",
        "BEGIN",
        "CALL",
        "COMMIT",
        "COPY",
        "DO",
        "EXPLAIN",
        "GRANT",
        "LISTEN",
        "NOTIFY",
        "PREPARE",
        "REINDEX",
        "REVOKE",
        "ROLLBACK",
        "SAVEPOINT",
        "SET",
        "VACUUM",
    }
)

# Keywords that turn a SELECT into a row lock, i.e. a write-intent statement.
_LOCKING_FOLLOWERS = frozenset({"UPDATE", "SHARE", "KEY", "NO"})

# Trailing semicolons, whitespace and comments, stripped before a LIMIT is
# appended. Appending after any of them produces either a second statement or a
# commented-out cap.
_TRAILING_NOISE = re.compile(r"(?:\s|;|--[^\n]*|/\*.*?\*/)+$", re.DOTALL)


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class GuardrailConfig:
    """Every rule, individually switchable, with defaults that are safe.

    The toggles exist so a caller can narrow the gate for a context that needs
    something unusual -- not so it can be opened wholesale. Turning several off
    at once will let writes through, which is the point of them being explicit.

    Parse validity is deliberately not switchable: without a parse tree there is
    nothing for any other rule to inspect.
    """

    max_rows: int = DEFAULT_MAX_ROWS
    max_subquery_depth: int = DEFAULT_MAX_SUBQUERY_DEPTH

    require_single_statement: bool = True
    require_select: bool = True
    block_forbidden_constructs: bool = True
    enforce_subquery_depth: bool = True
    enforce_row_limit: bool = True

    # When false, a query with no LIMIT is rejected instead of being capped.
    auto_limit: bool = True
    allow_comments: bool = False

    forbidden_functions: frozenset[str] = FORBIDDEN_FUNCTIONS
    forbidden_prefixes: tuple[str, ...] = FORBIDDEN_PREFIXES
    forbidden_keywords: frozenset[str] = FORBIDDEN_KEYWORDS


@dataclass(frozen=True)
class GuardrailResult:
    """The verdict. `rule` and `reason` are set together, on rejection only."""

    allowed: bool
    sql: str
    rule: str | None = None
    reason: str | None = None
    rewritten_sql: str | None = None

    @property
    def sql_to_execute(self) -> str:
        """The exact text an executor should run.

        Raises on a rejected result rather than returning the original SQL.
        `ClarificationNeeded` in generate.py makes the same trade: a caller that
        forgets to check `allowed` gets an exception, not a query nobody cleared.
        """
        if not self.allowed:
            raise RuntimeError(f"SQL was rejected by rule {self.rule!r}: {self.reason}")
        return self.rewritten_sql or self.sql


def _reject(sql: str, rule: str, reason: str) -> GuardrailResult:
    return GuardrailResult(allowed=False, sql=sql, rule=rule, reason=reason)


# ---------------------------------------------------------------- token walks


def _meaningful(statement: Statement) -> bool:
    """A statement with something in it besides whitespace and semicolons.

    `SELECT 1;;` parses as two statements, the second empty. Counting raw
    statements would reject it as multi-statement, which is wrong -- there is
    only one query there.
    """
    return bool(str(statement).strip().strip(";").strip())


def _visible(statement: Statement) -> list:
    """Flattened, whitespace dropped, so neighbours can be inspected by index."""
    return [tok for tok in statement.flatten() if not tok.is_whitespace]


def _identifier_text(token) -> str:
    """Lowercased name with any quoting removed, so `"LO_IMPORT"` still matches."""
    return token.value.strip('"').strip("`").lower()


def _select_bearing(paren: Parenthesis) -> bool:
    """True when this paren is a subquery, rather than a function call or list.

    Only its own tokens are checked, not nested ones: a SELECT two levels down
    belongs to that level's count, not this one's.
    """
    return any(
        tok.ttype is T.Keyword.DML and tok.value.upper() == "SELECT" for tok in paren.tokens
    )


def _subquery_depth(node: TokenList, depth: int = 0) -> int:
    deepest = depth
    for tok in node.tokens:
        if isinstance(tok, TokenList):
            nested = depth + 1 if isinstance(tok, Parenthesis) and _select_bearing(tok) else depth
            deepest = max(deepest, _subquery_depth(tok, nested))
    return deepest


def _is_limit(token) -> bool:
    return token.ttype is T.Keyword and token.value.upper() == "LIMIT"


def _top_level_row_cap(statement: Statement) -> tuple[bool, str | None]:
    """Whether the outermost query caps its rows, and the cap's literal text.

    Walks `statement.tokens` rather than `flatten()`, so a LIMIT belonging to a
    CTE body or an inline subquery does not count as the outer query's cap.
    Verified against plain, ORDER BY and CTE forms.
    """
    tokens = [tok for tok in statement.tokens if not tok.is_whitespace]
    for index, tok in enumerate(tokens):
        if _is_limit(tok):
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            return True, None if following is None else following.value
        if tok.ttype is T.Keyword and tok.value.upper() == "FETCH":
            return True, _fetch_count(tokens[index:])
    return False, None


def _fetch_count(tokens: list) -> str | None:
    """The n in `FETCH FIRST n ROWS ONLY`, or None when it is absent."""
    for tok in tokens[1:]:
        if tok.ttype is T.Literal.Number.Integer:
            return tok.value
        if tok.ttype is T.Keyword and tok.value.upper() not in {"FIRST", "NEXT"}:
            break
    return None


def _row_caps(statement: Statement) -> list[tuple[str, str | None]]:
    """(clause, literal) for every LIMIT and FETCH at any depth."""
    tokens = _visible(statement)
    caps: list[tuple[str, str | None]] = []
    for index, tok in enumerate(tokens):
        if _is_limit(tok):
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            caps.append(("LIMIT", None if following is None else following.value))
        elif tok.ttype is T.Keyword and tok.value.upper() == "FETCH":
            caps.append(("FETCH", _fetch_count(tokens[index:])))
    return caps


def _append_limit(sql: str, max_rows: int) -> str:
    """Cap an uncapped query at max_rows + 1.

    The extra row is not a loophole, it is the overflow signal. The executor
    reads one row past its own cap to report `truncated`; capped at exactly
    max_rows here, the database never has that row to give, and a query that
    matched 5000 rows comes back as a clean 1000 with truncated=False -- a
    silently partial answer. The executor still returns at most max_rows.

    The trailing-noise strip is what makes this correct: every few-shot example
    ends in a semicolon, and `... ORDER BY o.order_date DESC; LIMIT 1001` is two
    statements, the second of them nonsense. The newline matters for the same
    reason -- appended to the end of a line holding a `--` comment, the cap would
    be commented out.
    """
    return f"{_TRAILING_NOISE.sub('', sql)}\nLIMIT {max_rows + 1}"


# ----------------------------------------------------------------------- rules


def _check_parse(
    sql: str, config: GuardrailConfig
) -> tuple[Statement | None, GuardrailResult | None]:
    """Rule 1: one parseable statement, and only one."""
    statements = [stmt for stmt in sqlparse.parse(sql) if _meaningful(stmt)]

    if not statements:
        return None, _reject(sql, "parse", "no SQL statement found")

    if config.require_single_statement and len(statements) > 1:
        kinds = ", ".join(stmt.get_type() for stmt in statements)
        return None, _reject(
            sql,
            "single_statement",
            f"expected exactly one statement, found {len(statements)} ({kinds}); "
            "a trailing statement after a semicolon is never executed here",
        )

    statement = statements[0]

    errors = [tok.value for tok in statement.flatten() if tok.ttype in T.Error]
    if errors:
        return None, _reject(
            sql, "parse", f"SQL did not parse cleanly, unexpected {errors[0]!r}"
        )

    parens = [tok.value for tok in statement.flatten() if tok.ttype is T.Punctuation]
    opened, closed = parens.count("("), parens.count(")")
    if opened != closed:
        return None, _reject(
            sql, "parse", f"unbalanced parentheses, {opened} opened and {closed} closed"
        )

    return statement, None


def _check_statement_type(sql: str, statement: Statement) -> GuardrailResult | None:
    """Rule 2: SELECT, or WITH ... SELECT, and nothing else.

    sqlparse answers SELECT for both forms and UNKNOWN for EXPLAIN, COPY, DO,
    CALL, SET and unparseable text, so this one comparison rejects all of them.
    """
    kind = statement.get_type()
    if kind == "SELECT":
        return None

    leading = statement.token_first(skip_cm=True)
    if leading is not None and leading.ttype in T.Keyword:
        named = leading.value.upper()
    else:
        named = f"{kind} (no recognisable leading keyword)"
    return _reject(
        sql,
        "statement_type",
        f"only SELECT and WITH ... SELECT may be executed, got {named}",
    )


def _check_forbidden_constructs(
    sql: str, statement: Statement, config: GuardrailConfig
) -> GuardrailResult | None:
    """Rule 3: writes and escapes that hide inside a statement typed SELECT."""
    tokens = _visible(statement)

    for index, tok in enumerate(tokens):
        upper = tok.value.upper()

        # FOR UPDATE / FOR SHARE / FOR NO KEY UPDATE: a SELECT that takes a
        # write lock. Checked before the DML sweep so the message names the
        # locking clause rather than the bare UPDATE token inside it.
        if tok.ttype is T.Keyword and upper == "FOR":
            clause = []
            for following in tokens[index + 1 :]:
                if following.value.upper() not in _LOCKING_FOLLOWERS:
                    break
                clause.append(following.value.upper())
            if clause:
                return _reject(
                    sql,
                    "forbidden_construct",
                    f"row-locking clause FOR {' '.join(clause)} is not a read",
                )

        if tok.ttype is T.Keyword.DML and upper != "SELECT":
            return _reject(
                sql,
                "forbidden_construct",
                f"{upper} writes data; a data-modifying CTE is still a write "
                "even when the statement as a whole reads as a SELECT",
            )

        if tok.ttype is T.Keyword.DDL:
            return _reject(sql, "forbidden_construct", f"{upper} changes schema")

        if tok.ttype is T.Keyword.DCL:
            return _reject(sql, "forbidden_construct", f"{upper} changes privileges")

        if tok.ttype is T.Keyword and upper == "INTO":
            return _reject(
                sql, "forbidden_construct", "INTO writes the result set to a new table"
            )

        if tok.ttype is T.Keyword and upper in config.forbidden_keywords:
            return _reject(sql, "forbidden_construct", f"{upper} is not permitted")

        # String.Symbol is a double-quoted identifier, not a string literal --
        # `"lo_import"('/etc/passwd')` runs, so the quoted form has to be matched
        # too. Single-quoted literals are String.Single and stay untouched.
        if tok.ttype in T.Name or tok.ttype is T.Literal.String.Symbol:
            name = _identifier_text(tok)
            if name in config.forbidden_functions or name.startswith(config.forbidden_prefixes):
                return _reject(
                    sql,
                    "forbidden_construct",
                    f"{name}() can read files, reach another host or stall the "
                    "connection without needing a write privilege",
                )

    return None


def _check_subquery_depth(
    sql: str, statement: Statement, config: GuardrailConfig
) -> GuardrailResult | None:
    """Rule 4: nesting, counting only parens that actually hold a subquery."""
    depth = _subquery_depth(statement)
    if depth > config.max_subquery_depth:
        return _reject(
            sql,
            "subquery_depth",
            f"subqueries nested {depth} deep, limit is {config.max_subquery_depth}",
        )
    return None


def _check_row_limit(
    sql: str, statement: Statement, config: GuardrailConfig
) -> GuardrailResult | None:
    """Rule 5: reject an oversized cap, add one when the outer query has none.

    The size check covers every depth -- a CTE that materialises a million rows
    is expensive whatever the outer query then selects from it -- while the
    rewrite looks only at the outermost query, which is the one whose row count
    reaches this process.

    The ceiling is max_rows + 1, not max_rows, because that is the cap
    `_append_limit` writes: a rewritten query has to pass its own re-check. The
    extra row never reaches a caller -- the executor returns at most max_rows
    and uses the one beyond it only to report truncation.
    """
    ceiling = config.max_rows + 1
    for clause, literal in _row_caps(statement):
        if literal is None:
            return _reject(sql, "row_limit", f"{clause} with no value")
        if literal.upper() == "ALL":
            return _reject(sql, "row_limit", "LIMIT ALL removes the row cap")
        if not literal.isdigit():
            return _reject(
                sql,
                "row_limit",
                f"{clause} value {literal!r} is not a literal integer, so it "
                "cannot be verified against the row cap",
            )
        if int(literal) > ceiling:
            return _reject(
                sql,
                "row_limit",
                f"{clause} {literal} exceeds the maximum of {config.max_rows} rows",
            )
    return None


def _check_comments(
    sql: str, statement: Statement, config: GuardrailConfig
) -> GuardrailResult | None:
    """Rule 6: no comments.

    Comments carry no meaning for the database and a great deal for anyone
    trying to hide a clause from a reviewer, so they are refused rather than
    stripped. Refusing is visible; stripping silently changes what was asked.
    """
    if config.allow_comments:
        return None
    for tok in statement.flatten():
        if tok.ttype in T.Comment:
            snippet = " ".join(tok.value.split())[:40]
            return _reject(sql, "comment", f"comments are not allowed: {snippet!r}")
    return None


# ------------------------------------------------------------------- the gate


def check(sql: str, config: GuardrailConfig | None = None) -> GuardrailResult:
    """Decide whether `sql` may be executed, and with what row cap.

    Rules run in a fixed order and the first failure returns, so the reported
    rule is the most fundamental thing wrong rather than an incidental
    consequence of it. A query that both parses as two statements and contains a
    DROP is reported as multi-statement, because that is what has to be fixed
    first.

    Returns a result whose `rewritten_sql` is set only when a LIMIT was added.
    """
    config = config or GuardrailConfig()

    statement, failure = _check_parse(sql, config)
    if failure is not None:
        return failure
    assert statement is not None  # _check_parse returns one or the other

    if config.require_select:
        failure = _check_statement_type(sql, statement)
        if failure is not None:
            return failure

    if config.block_forbidden_constructs:
        failure = _check_forbidden_constructs(sql, statement, config)
        if failure is not None:
            return failure

    if config.enforce_subquery_depth:
        failure = _check_subquery_depth(sql, statement, config)
        if failure is not None:
            return failure

    rewritten: str | None = None
    if config.enforce_row_limit:
        failure = _check_row_limit(sql, statement, config)
        if failure is not None:
            return failure

        capped, _ = _top_level_row_cap(statement)
        if not capped:
            if not config.auto_limit:
                return _reject(
                    sql,
                    "row_limit",
                    f"no LIMIT present and auto_limit is off; add LIMIT {config.max_rows} or less",
                )
            rewritten = _append_limit(sql, config.max_rows)

    failure = _check_comments(sql, statement, config)
    if failure is not None:
        return failure

    return GuardrailResult(allowed=True, sql=sql, rewritten_sql=rewritten)
