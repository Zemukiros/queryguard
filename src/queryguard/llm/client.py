"""Thin wrapper over the Anthropic SDK: request cap, cost accounting, call log.

Three things this owns that the SDK does not:

1. A hard request cap. An agent loop that misbehaves can spend real money very
   quickly, so the counter is process-global rather than per-instance -- a
   runaway loop's natural failure mode is constructing a fresh client, which
   would reset an instance counter and defeat the guard entirely.
2. Accurate cost. Cached tokens are billed at different rates from fresh ones
   (1.25x to write, 0.1x to read), so a naive input+output calculation
   understates a cache-miss call and overstates every cache hit after it.
3. A durable JSONL log, so `running_total_usd()` survives process restarts.

Model IDs and prices below are the current published values and are complete as
written -- do NOT append a date suffix to a model ID; that produces a 404.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from queryguard.config import REPO_ROOT, load_env

DEFAULT_MODEL = "claude-sonnet-5"

# Runaway-loop guard. Override with QUERYGUARD_MAX_REQUESTS.
DEFAULT_MAX_REQUESTS = 50

# Cache multipliers apply to the model's *input* rate.
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL (a 1-hour TTL would be 2.0x)
CACHE_READ_MULTIPLIER = 0.10


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float


PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-5": ModelPricing(input_usd_per_mtok=2.00, output_usd_per_mtok=10.00),
    "claude-haiku-4-5": ModelPricing(input_usd_per_mtok=1.00, output_usd_per_mtok=5.00),
}


class RequestCapExceeded(RuntimeError):
    """The process-wide request cap was hit. No API call was made."""


# Module-level on purpose: see the docstring. Not thread-safe by design -- the
# guard is against runaway loops, not concurrent workers.
_request_count = 0


def request_count() -> int:
    return _request_count


def reset_request_count() -> None:
    """Reset the guard. Intended for tests; never call this inside a loop."""
    global _request_count
    _request_count = 0


def max_requests() -> int:
    """Read at call time so the env var can be set after import."""
    raw = os.getenv("QUERYGUARD_MAX_REQUESTS")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_REQUESTS
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"QUERYGUARD_MAX_REQUESTS is not an integer: {raw!r}") from exc


def log_path() -> Path:
    override = os.getenv("QUERYGUARD_LLM_LOG")
    return Path(override) if override else REPO_ROOT / "logs" / "llm_calls.jsonl"


# ------------------------------------------------------------------- costing


def _usage_field(usage: Any, name: str) -> int:
    """Usage fields are optional in the SDK model; absent means zero."""
    return int(getattr(usage, name, None) or 0)


def estimate_cost_usd(model: str, usage: Any) -> float:
    """Cost of one call, with cache-written and cache-read tokens priced apart.

    `usage.input_tokens` already excludes cached tokens -- the API reports the
    three input buckets separately -- so these terms do not double-count.
    """
    pricing = PRICING.get(model)
    if pricing is None:
        # Silently costing $0 for an unrecognised model would corrupt the
        # running total, which is the one number this module exists to get right.
        raise KeyError(f"No pricing for model {model!r}; known: {sorted(PRICING)}")

    per_input_token = pricing.input_usd_per_mtok / 1_000_000
    per_output_token = pricing.output_usd_per_mtok / 1_000_000

    return (
        _usage_field(usage, "input_tokens") * per_input_token
        + _usage_field(usage, "cache_creation_input_tokens")
        * per_input_token
        * CACHE_WRITE_MULTIPLIER
        + _usage_field(usage, "cache_read_input_tokens")
        * per_input_token
        * CACHE_READ_MULTIPLIER
        + _usage_field(usage, "output_tokens") * per_output_token
    )


def prompt_hash(system_blocks: Any, user_message: str) -> str:
    """Stable fingerprint of a prompt. The text itself is never logged."""
    payload = json.dumps(
        {"system": system_blocks, "user": user_message}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- logging


def append_log(entry: dict[str, Any], path: Path | None = None) -> Path:
    target = path or log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return target


def running_total_usd(path: Path | None = None) -> float:
    """Sum estimated_cost_usd over the log.

    A partially written final line (killed mid-append) is skipped rather than
    raised on: a corrupt tail should not make the cost report unavailable.
    """
    target = path or log_path()
    if not target.is_file():
        return 0.0

    total = 0.0
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            total += float(json.loads(line).get("estimated_cost_usd", 0.0))
        except (ValueError, TypeError):
            continue
    return total


# -------------------------------------------------------------------- client


@dataclass
class CallResult:
    parsed: Any
    message: Any
    usage: Any
    cost_usd: float
    latency_ms: int


class LLMClient:
    """Anthropic SDK wrapper. Pass `sdk_client` to inject a fake in tests."""

    def __init__(self, sdk_client: Any = None, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        if sdk_client is not None:
            self._client = sdk_client
        else:
            load_env()
            import anthropic

            # The SDK's own retry policy is exactly what we want and no more:
            # it retries 408/409/429/5xx and connection errors, and never
            # retries 400/401/403/404. Stated explicitly so nobody "improves"
            # it into a loop that hammers the API on a malformed request.
            self._client = anthropic.Anthropic(max_retries=2)

    def complete(
        self,
        system_blocks: list[dict[str, Any]],
        user_message: str,
        output_format: Any,
        *,
        max_tokens: int = 4096,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        log: Path | None = None,
    ) -> CallResult:
        """One structured-output call, capped, costed and logged."""
        global _request_count

        cap = max_requests()
        if _request_count >= cap:
            # Raise before incrementing and before any network call, so the cap
            # is a real ceiling on API requests rather than on attempts.
            raise RequestCapExceeded(
                f"request cap of {cap} reached ({_request_count} made); "
                "raise QUERYGUARD_MAX_REQUESTS if this is intentional"
            )
        _request_count += 1

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_message}],
            "output_format": output_format,
        }
        if thinking is not None:
            kwargs["thinking"] = thinking
        if output_config is not None:
            kwargs["output_config"] = output_config

        started = time.perf_counter()
        message = self._client.messages.parse(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = message.usage
        cost = estimate_cost_usd(self.model, usage)

        append_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "input_tokens": _usage_field(usage, "input_tokens"),
                "output_tokens": _usage_field(usage, "output_tokens"),
                "cache_creation_input_tokens": _usage_field(
                    usage, "cache_creation_input_tokens"
                ),
                "cache_read_input_tokens": _usage_field(usage, "cache_read_input_tokens"),
                "estimated_cost_usd": cost,
                "latency_ms": latency_ms,
                "prompt_sha256": prompt_hash(system_blocks, user_message),
            },
            log,
        )

        return CallResult(
            parsed=message.parsed_output,
            message=message,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
