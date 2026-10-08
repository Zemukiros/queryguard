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

The cap and the log sit behind `CallGuard`. The default guard is the process cap
and the JSONL file, which is what the CLI, the evals and tests use. The API
passes its own guard (queryguard.state): a daily call cap and a spend ledger
shared by every instance, because a per-process counter means nothing when a
serverless platform runs twenty processes and recycles them at will.

Model IDs and prices below are the current published values and are complete as
written -- do NOT append a date suffix to a model ID; that produces a 404.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from queryguard.config import load_env, log_target

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"

# Runaway-loop guard. Override with QUERYGUARD_MAX_REQUESTS.
DEFAULT_MAX_REQUESTS = 50

# The most API calls one question may make (pipeline.py enforces it).
MAX_CALLS_PER_QUESTION = 4

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
    # The minimum cacheable prefix is per-model: 1024 tokens on Sonnet 5, 4096
    # on Haiku 4.5. build_system_blocks() marks 4,953 tokens -- measured on a
    # live call, not estimated from character count, which understates it by
    # about half because the schema block tokenizes at roughly 2 chars/token.
    # So the prefix clears Haiku's floor as well, but by only ~21%, and the
    # schema block is most of it: introspect a smaller database, or profile
    # fewer columns, and the prefix slips under 4096 while still caching on
    # Sonnet. That failure is silent -- no error, no cache entry, just every
    # call billed at the full input rate -- which is why the runtime check in
    # _warn_if_cache_was_ignored() stays even though today's prefix is fine.
    "claude-haiku-4-5": ModelPricing(input_usd_per_mtok=1.00, output_usd_per_mtok=5.00),
}


def sdk_error_types() -> tuple[type[BaseException], ...]:
    """(anthropic.APIError,) once the SDK is loaded, else ().

    Callers that classify errors use this instead of importing anthropic, so a
    process that never builds a real client (demo mode, /healthz) never pays
    for the import. If the SDK was never loaded, none of its errors can occur.
    """
    sdk = sys.modules.get("anthropic")
    return (sdk.APIError,) if sdk is not None else ()


def is_auth_failure(exc: BaseException | None) -> bool:
    """The API refused the key: missing, expired or revoked (401), or not allowed (403)."""
    sdk = sys.modules.get("anthropic")
    return sdk is not None and isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError))


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
    # On Vercel the API's real ledger is Redis (state/redis.py), so the file is off.
    return log_target("QUERYGUARD_LLM_LOG", "llm_calls.jsonl", on_platform="off")


class CallGuard:
    """Admits each API call before it is made and records it afterwards.

    This default is the process-wide request cap plus the JSONL log.
    """

    def before_call(self) -> None:
        """Raise RequestCapExceeded to refuse the call. Runs before any network I/O."""
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

    def after_call(self, entry: dict[str, Any], log: Path | None = None) -> None:
        """Record a made call: its cost (estimated_cost_usd) and log fields."""
        append_log(entry, log)


PROCESS_GUARD = CallGuard()


class DemoCallGuard(CallGuard):
    """For the simulated model: never capped, never counted as spend.

    Calls go to their own log (or none), so the real ledger -- and the spend
    ceiling read from it -- never sees them.
    """

    def __init__(self, log: Path | None = None) -> None:
        self._log = log

    def before_call(self) -> None:
        pass

    def after_call(self, entry: dict[str, Any], log: Path | None = None) -> None:
        if self._log is not None:
            append_log(entry, self._log)


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


# Deliberately pessimistic: two characters per token undercounts characters,
# so it overcounts tokens -- the prompt measured ~2 chars/token at its densest.
_CHARS_PER_TOKEN_LOW = 2


def upper_bound_cost_usd(model: str, system_blocks: Any, user_message: str, max_tokens: int) -> float:
    """What a call whose usage was never returned could have cost, at most.

    Every input token priced at the full (uncached) rate, and the full
    max_tokens of output.
    """
    pricing = PRICING[model]
    text = json.dumps(system_blocks, default=str) + user_message
    input_tokens = len(text) / _CHARS_PER_TOKEN_LOW
    return (
        input_tokens * pricing.input_usd_per_mtok + max_tokens * pricing.output_usd_per_mtok
    ) / 1_000_000


def _was_answered(exc: Exception) -> bool:
    """Whether the API produced a response the SDK then failed on.

    A response that fails validation was generated and billed. A connection
    error or a 4xx was not; a 429 or 5xx is not billed either. Unknown errors
    count as billed, so the log errs towards overcounting.
    """
    import anthropic

    if isinstance(exc, anthropic.APIConnectionError):
        return False
    if isinstance(exc, anthropic.APIStatusError):
        return False
    return True


def _cache_was_requested(system_blocks: Any) -> bool:
    """True when at least one system block carries a cache_control marker."""
    if not isinstance(system_blocks, list):
        return False
    return any(isinstance(block, dict) and "cache_control" in block for block in system_blocks)


def _warn_if_cache_was_ignored(model: str, system_blocks: Any, usage: Any) -> None:
    """Warn when a marked prefix neither wrote nor read cache.

    Both counters at zero is the only symptom the API offers: a prefix below the
    model's minimum cacheable length is not an error, it is silently not cached.
    The request succeeds, nothing in the response mentions the breakpoint, and
    the only other evidence is the input bill.
    """
    if not _cache_was_requested(system_blocks):
        return
    if _usage_field(usage, "cache_creation_input_tokens") or _usage_field(
        usage, "cache_read_input_tokens"
    ):
        return

    logger.warning(
        "cache_control was set but %s neither wrote nor read cache "
        "(cache_creation_input_tokens=0, cache_read_input_tokens=0); the marked "
        "prefix is most likely below this model's minimum cacheable length",
        model,
    )


def prompt_hash(system_blocks: Any, user_message: str) -> str:
    """Stable fingerprint of a prompt. The text itself is never logged."""
    payload = json.dumps(
        {"system": system_blocks, "user": user_message}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- logging


def append_log(entry: dict[str, Any], path: Path | None = None) -> Path:
    """Append one JSON line. A path of "-" prints it to stdout; "off" drops it."""
    target = path or log_path()
    line = json.dumps(entry, sort_keys=True)
    if str(target) == "off":
        return target
    if str(target) == "-":
        print(line, flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
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

    def __init__(self, sdk_client: Any = None, model: str = DEFAULT_MODEL, guard: CallGuard | None = None) -> None:
        self.model = model
        self.guard = guard or PROCESS_GUARD
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
        """One structured-output call, capped, costed and logged (see CallGuard)."""
        self.guard.before_call()

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
        try:
            message = self._client.messages.parse(**kwargs)
        except Exception as exc:
            # The request was made, and probably billed, but the response
            # failed to parse or the request failed, and the SDK raised before
            # handing back usage. Log it anyway, priced as an upper bound, or
            # every cost total built on this log undercounts. The first eval
            # run lost four calls this way.
            if _was_answered(exc):
                self.guard.after_call(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": self.model,
                        "usage_unknown": True,
                        "error": type(exc).__name__,
                        "estimated_cost_usd": upper_bound_cost_usd(
                            self.model, system_blocks, user_message, max_tokens
                        ),
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "prompt_sha256": prompt_hash(system_blocks, user_message),
                    },
                    log,
                )
            raise
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = message.usage
        _warn_if_cache_was_ignored(self.model, system_blocks, usage)
        cost = estimate_cost_usd(self.model, usage)

        self.guard.after_call(
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
