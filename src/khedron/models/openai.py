from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from khedron.budget import BudgetReserver, upper_bound_cost_usd
from khedron.cost.pricing import compute_cost_usd, load_vendor_pricing
from khedron.errors import (
    ModelAuthenticationError,
    ModelBillingError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from khedron.models.base import AnswerModel
from khedron.models.registry import register_model
from khedron.pacing import (
    TIMEOUT_MAX_RETRIES,
    DispatchOutcome,
    RetryDisposition,
    estimate_reservation_tokens,
    paced_generate,
    retry_after_seconds,
)
from khedron.ratelimit import RateLimiter
from khedron.types import APICallResult

__all__ = ["OpenAIModel", "OpenAIModelConfig"]

JsonObject: TypeAlias = dict[str, Any]

_OPENAI_VENDOR = "openai"


class _ResponsesAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI

    async def close(self) -> None: ...


class _ClientFactory(Protocol):
    def __call__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        timeout_seconds: float,
    ) -> _OpenAIClient: ...


class OpenAIModelConfig(BaseModel):
    """Configuration for the OpenAI answer model adapter."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1)
    api_key_env_var: str = Field(default="OPENAI_API_KEY", min_length=1)
    base_url: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


@register_model("openai")
class OpenAIModel(AnswerModel):
    """OpenAI Responses API-backed answer model."""

    def __init__(
        self,
        config: OpenAIModelConfig,
        *,
        client_factory: _ClientFactory | None = None,
        backoff_sleep: Callable[[float], Awaitable[None]] | None = None,
        rate_limiter: RateLimiter | None = None,
        rate_limit_scope: str | None = None,
        budget: BudgetReserver | None = None,
    ) -> None:
        self._config = config
        self._client_factory = (
            client_factory if client_factory is not None else _create_openai_client
        )
        self._backoff_sleep = backoff_sleep if backoff_sleep is not None else asyncio.sleep
        # One limiter shared across every paid call in a run, or None to leave behaviour untouched.
        # The scope is the raw config id; it is vendor-qualified where the reservation is made.
        self._rate_limiter = rate_limiter
        self._rate_limit_scope = rate_limit_scope
        # The one shared money ledger, injected the same way as the limiter; None leaves the spend
        # path exactly as before (no pre-dispatch reservation, cap enforced post-hoc only).
        self._budget = budget
        self._client: _OpenAIClient | None = None
        self._pricing = load_vendor_pricing(_OPENAI_VENDOR)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def vendor(self) -> str:
        return _OPENAI_VENDOR

    async def initialize(self) -> None:
        if self._client is not None:
            return

        api_key = os.environ.get(self._config.api_key_env_var)
        if api_key is None or not api_key.strip():
            raise ModelAuthenticationError(
                "OpenAI API key is not configured.",
                api_key_env_var=self._config.api_key_env_var,
            )

        try:
            self._client = self._client_factory(
                api_key=api_key,
                base_url=self._config.base_url,
                timeout_seconds=self._config.timeout_seconds,
            )
        except Exception as exc:
            raise ModelError("Failed to create OpenAI client.") from exc

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> APICallResult:
        await self.initialize()
        client = self._require_client()
        started_at = time.perf_counter()

        async def dispatch() -> object:
            return await client.responses.create(
                model=self._config.model_id,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )

        def extract(response: object) -> APICallResult:
            return self._result_from_response(
                response, latency_ms=(time.perf_counter() - started_at) * 1000
            )

        return await paced_generate(
            limiter=self._rate_limiter,
            vendor=_OPENAI_VENDOR,
            model_id=self._config.model_id,
            scope=self._rate_limit_scope,
            estimate=lambda: estimate_reservation_tokens(
                vendor=_OPENAI_VENDOR,
                model_id=self._config.model_id,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
            ),
            max_retries=self._config.max_retries,
            dispatch=dispatch,
            extract=extract,
            classify=self._classify,
            backoff_sleep=self._backoff_sleep,
            budget=self._budget,
            estimate_cost=lambda: self._estimate_max_cost_usd(prompt, max_output_tokens),
        )

    def _estimate_max_cost_usd(self, prompt: str, max_output_tokens: int) -> float:
        """The hard upper-bound cost of one attempt, or a fail-closed refusal if unpriceable.

        Only ever called when a budget ledger is injected (the pacer gates it on ``budget``), so an
        uncapped run never prices pre-dispatch and behaves exactly as before. A model with no usable
        price is refused here, before dispatch, rather than billed then discovered post-hoc as it
        is today.
        """
        try:
            return upper_bound_cost_usd(
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                model_id=self._config.model_id,
                pricing=self._pricing,
            )
        except ValueError as exc:
            raise ModelError(
                "Pricing is not configured for this model, so its cost cannot be bounded before "
                "dispatch; refusing rather than billing an unpriceable call.",
                model_id=self._config.model_id,
            ) from exc

    def _classify(self, exc: Exception) -> DispatchOutcome:
        """Map a dispatch exception to a disposition, preserving the prior classification exactly.

        Only the loop moved to the pacer; which errors are terminal, retryable, or rate limits is
        unchanged. Timeout, auth and billing stay terminal; a 429 is a rate limit; server and
        transient connection failures are retryable; anything else is a terminal generic failure.
        """
        # A ModelError raised by the client is already classified; re-raise it unchanged, as the
        # prior `except ModelError: raise` did. Without this a ModelRateLimitError bubbling up would
        # be reclassified and retried, and its hold released.
        if isinstance(exc, ModelError):
            return DispatchOutcome(RetryDisposition.TERMINAL, lambda _n: exc)
        model_id = self._config.model_id
        if _is_timeout_error(exc):
            # Retryable, but with the timeout class's own hard cap (TIMEOUT_MAX_RETRIES) so a
            # duplicate-billing-prone timeout cannot inherit the suite's larger max_retries. On
            # exhaustion the surfaced error is stamped retryable=True (RETRYABLE is not TERMINAL),
            # making a persistently timed-out question auto-eligible for stage recovery on resume.
            return DispatchOutcome(
                RetryDisposition.RETRYABLE,
                lambda _n: ModelTimeoutError("OpenAI model request timed out.", model_id=model_id),
                retry_cap=TIMEOUT_MAX_RETRIES,
            )
        if _is_authentication_error(exc):
            return DispatchOutcome(
                RetryDisposition.TERMINAL,
                lambda _n: ModelAuthenticationError(
                    "OpenAI authentication failed.", model_id=model_id
                ),
            )
        if _is_billing_error(exc):
            reason = _error_reason(exc)
            return DispatchOutcome(
                RetryDisposition.TERMINAL,
                lambda _n: ModelBillingError(
                    "OpenAI billing or quota is unavailable.", model_id=model_id, reason=reason
                ),
            )
        kind = _retryable_error_kind(exc)
        if kind == "rate_limit":
            return DispatchOutcome(
                RetryDisposition.RATE_LIMITED,
                lambda n: ModelRateLimitError(
                    "OpenAI rate limit persisted after retries.", model_id=model_id, attempts=n
                ),
                retry_after=retry_after_seconds(exc),
            )
        if kind is not None:
            return DispatchOutcome(
                RetryDisposition.RETRYABLE,
                lambda n: ModelError("OpenAI model request failed.", model_id=model_id, attempts=n),
            )
        return DispatchOutcome(
            RetryDisposition.TERMINAL,
            lambda n: ModelError("OpenAI model request failed.", model_id=model_id, attempts=n),
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    def _require_client(self) -> _OpenAIClient:
        if self._client is None:
            raise ModelError("OpenAIModel is not initialized.")
        return self._client

    def _result_from_response(self, response: object, *, latency_ms: float) -> APICallResult:
        output = _extract_output_text(response)
        input_tokens = _extract_token_count(response, ("input_tokens", "prompt_tokens"))
        output_tokens = _extract_token_count(response, ("output_tokens", "completion_tokens"))
        cached_input_tokens = _extract_cached_input_tokens(response)

        try:
            cost_usd = compute_cost_usd(
                self._config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=self._pricing,
            )
        except ValueError as exc:
            raise ModelError(
                "OpenAI pricing is not configured for the requested model.",
                model_id=self._config.model_id,
            ) from exc

        return APICallResult(
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model_id=self._config.model_id,
            raw_response=_compact_raw_response(response),
        )


def _create_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    timeout_seconds: float,
) -> _OpenAIClient:
    from openai import AsyncOpenAI

    kwargs: JsonObject = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    return cast(_OpenAIClient, AsyncOpenAI(**kwargs))


def _extract_output_text(response: object) -> str:
    output_text = _field(response, "output_text")
    if isinstance(output_text, str):
        return output_text
    raise ModelError("OpenAI response did not include output text.")


def _extract_token_count(response: object, names: tuple[str, ...]) -> int:
    usage = _field(response, "usage")
    for name in names:
        count = _field(usage, name)
        if isinstance(count, bool):
            continue
        if isinstance(count, int) and count >= 0:
            return count
    raise ModelError("OpenAI response did not include token usage.")


def _extract_cached_input_tokens(response: object) -> int:
    """Prompt tokens OpenAI served from cache (usage.{input,prompt}_tokens_details.cached_tokens).

    Unlike the required input/output counts this is not fail-closed: caching is optional and older
    responses omit the details object, so an absent or malformed value yields 0. It is captured for
    observability only and never discounts cost -- the pricing schema excludes cached pricing, so
    the recorded cost stays full-price and the budget cap stays conservative.
    """
    cached = _extract_cached_input_tokens_from_usage(_field(response, "usage"))
    return cached if cached is not None else 0


def _compact_raw_response(response: object) -> JsonObject:
    raw: JsonObject = {}
    for name in ("id", "model", "created_at", "object"):
        value = _field(response, name)
        if _is_json_scalar(value):
            raw[name] = value

    output_text = _field(response, "output_text")
    if isinstance(output_text, str):
        raw["output_text"] = output_text

    usage = _field(response, "usage")
    usage_raw = _compact_usage(usage)
    if usage_raw:
        raw["usage"] = usage_raw
    return raw


def _compact_usage(usage: object) -> JsonObject:
    raw: JsonObject = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        value = _field(usage, name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            raw[name] = value
    cached = _extract_cached_input_tokens_from_usage(usage)
    if cached is not None:
        raw["cached_input_tokens"] = cached
    return raw


def _usage_input_token_count(usage: object) -> int | None:
    """The prompt-token count from a usage object, non-raising (unlike _extract_token_count)."""
    for name in ("input_tokens", "prompt_tokens"):
        count = _field(usage, name)
        if isinstance(count, bool):
            continue
        if isinstance(count, int) and count >= 0:
            return count
    return None


def _extract_cached_input_tokens_from_usage(usage: object) -> int | None:
    for details_name in ("input_tokens_details", "prompt_tokens_details"):
        details = _field(usage, details_name)
        count = _field(details, "cached_tokens")
        if isinstance(count, bool):
            continue
        if isinstance(count, int) and count >= 0:
            # cached_tokens is a subset of the prompt tokens; a value above the reported input is
            # malformed, so drop it rather than persist an impossible count.
            input_count = _usage_input_token_count(usage)
            if input_count is not None and count > input_count:
                return None
            return count
    return None


def _field(source: object, name: str) -> object:
    if source is None:
        return None
    if isinstance(source, Mapping):
        value = cast(Mapping[str, object], source).get(name)
        return value
    return getattr(source, name, None)


def _is_json_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


def _is_timeout_error(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def _is_authentication_error(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code in {401, 403}:
        return True
    name = type(exc).__name__.lower()
    return "authentication" in name or "permissiondenied" in name


def _is_billing_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(
        marker in text
        for marker in (
            "billing",
            "credit balance",
            "credits",
            "insufficient_quota",
            "payment required",
            "purchase credits",
            "quota exceeded",
            "resource_exhausted",
            "too low to access",
        )
    )


def _error_reason(exc: Exception) -> str:
    text = _error_text(exc)
    for marker in (
        "insufficient_quota",
        "credit balance",
        "quota exceeded",
        "billing",
        "resource_exhausted",
    ):
        if marker in text:
            return marker
    return type(exc).__name__


def _error_text(exc: Exception) -> str:
    parts = [type(exc).__name__, str(exc)]
    for name in ("code", "type"):
        value = _field(exc, name)
        if isinstance(value, str):
            parts.append(value)
    body = _field(exc, "body")
    if body is not None:
        parts.append(str(body))
    error = _field(body, "error")
    if error is not None:
        parts.append(str(error))
        for name in ("code", "type", "message"):
            value = _field(error, name)
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts).lower()


def _retryable_error_kind(exc: Exception) -> str | None:
    status_code = _status_code(exc)
    name = type(exc).__name__.lower()
    if status_code == 429 or "ratelimit" in name or "rate_limit" in name:
        return "rate_limit"
    if status_code is not None and 500 <= status_code <= 599:
        return "server"
    if "internalserver" in name or "serviceunavailable" in name:
        return "server"
    # Transient connection failures never reach an HTTP response, so they carry no
    # status code: the SDK surfaces httpx.ConnectError as APIConnectionError. Over a
    # long benchmark of thousands of calls a brief network blip is expected, and it
    # must not permanently fail a question — retry it with the same backoff. Auth,
    # billing, and timeout are classified before this and stay terminal.
    if status_code is None and ("connection" in name or "connecterror" in name):
        return "connection"
    return None


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, bool):
        return None
    if isinstance(status_code, int):
        return status_code
    return None
