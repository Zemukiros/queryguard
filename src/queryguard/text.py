"""Normalisation shared by the API cache and the demo fake."""

from __future__ import annotations

import unicodedata


def normalize_question(question: str) -> str:
    """Case, spacing and trailing punctuation do not change what is asked."""
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(text.split()).rstrip(" ?.!")


def normalize_sql(sql: str) -> str:
    """Whitespace and a trailing semicolon do not change a query."""
    return " ".join(sql.split()).rstrip("; ").casefold()
