"""One short id for everything the models are told.

An eval number is only meaningful next to the prompt that produced it. This
hashes every instruction text, the few-shot examples, the rendered schema
(column comments included, so the glossary on orders.total_amount counts) and
the model ids. Eval rows record it; docs/EVAL_RESULTS.md quotes it. Any edit to
any of them gives a new id.
"""

from __future__ import annotations

import hashlib

from queryguard.llm.client import DEFAULT_MODEL
from queryguard.llm.examples import render_examples
from queryguard.llm.prompt import SYSTEM_INSTRUCTIONS
from queryguard.schema.introspect import DatabaseSchema
from queryguard.validation.agreement import SECOND_OPINION_INSTRUCTION
from queryguard.validation.backtranslate import BACKTRANSLATE_INSTRUCTIONS, JUDGE_GLOSSARY, JUDGE_INSTRUCTIONS, VALIDATION_MODEL


def prompt_version(schema: DatabaseSchema) -> str:
    parts = (
        DEFAULT_MODEL, VALIDATION_MODEL, SYSTEM_INSTRUCTIONS, render_examples(), schema.render_for_prompt(),
        SECOND_OPINION_INSTRUCTION, BACKTRANSLATE_INSTRUCTIONS, JUDGE_INSTRUCTIONS, JUDGE_GLOSSARY,
    )
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"p-{digest[:12]}"
