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

__all__ = ["AnthropicModel", "AnthropicModelConfig"]

JsonObject: TypeAlias = dict[str, Any]

_ANTHROPIC_VENDOR = "anthropic"


class _MessagesAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _AnthropicClient(Protocol):
    messages: _MessagesAPI

    async def close(self) -> None: ...


class _ClientFactory(Protocol):
    def __call__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> _AnthropicClient: ...


class AnthropicModelConfig(BaseModel):
    """Configuration for the Anthropic answer model adapter."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1)
    api_key_env_var: str = Field(default="ANTHROPIC_API_KEY", min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


@register_model("anthropic")
class AnthropicModel(AnswerModel):
    """Anthropic Messages API-backed answer model."""

    def __init__(
        self,
        config: AnthropicModelConfig,
        *,
        client_factory: _ClientFactory | None = None,
        backoff_sleep: Callable[[float], Awaitable[None]] | None = None,
        rate_limiter: RateLimiter | None = None,
        rate_limit_scope: str | None = None,
        budget: BudgetReserver | None = None,
    ) -> None:
        self._config = config
        self._client_factory = (
            client_factory if client_factory is not None else _create_anthropic_client
        )
        self._backoff_sleep = backoff_sleep if backoff_sleep is not None else asyncio.sleep
        # One limiter shared across every paid call in a run, or None to leave behaviour untouched.
        # The scope is the raw config id; it is vendor-qualified where the reservation is made.
        self._rate_limiter = rate_limiter
        self._rate_limit_scope = rate_limit_scope
        # The one shared money ledger, injected the same way as the limiter; None leaves the spend
        # path exactly as before (no pre-dispatch reservation, cap enforced post-hoc only).
        self._budget = budget
        self._client: _AnthropicClient | None = None
        self._pricing = load_vendor_pricing(_ANTHROPIC_VENDOR)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def vendor(self) -> str:
        return _ANTHROPIC_VENDOR

    async def initialize(self) -> None:
        if self._client is not None:
            return

        api_key = os.environ.get(self._config.api_key_env_var)
        if api_key is None or not api_key.strip():
            raise ModelAuthenticationError(
                "Anthropic API key is not configured.",
                api_key_env_var=self._config.api_key_env_var,
            )

        try:
            self._client = self._client_factory(
                api_key=api_key,
                timeout_seconds=self._config.timeout_seconds,
            )
        except Exception as exc:
            raise ModelError("Failed to create Anthropic client.") from exc

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
            return await client.messages.create(
                model=self._config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_output_tokens,
                temperature=temperature,
            )

        def extract(response: object) -> APICallResult:
            return self._result_from_response(
                response, latency_ms=(time.perf_counter() - started_at) * 1000
            )

        return await paced_generate(
            limiter=self._rate_limiter,
            vendor=_ANTHROPIC_VENDOR,
            model_id=self._config.model_id,
            scope=self._rate_limit_scope,
            estimate=lambda: estimate_reservation_tokens(
                vendor=_ANTHROPIC_VENDOR,
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
        price is refused here, before dispatch, rather than billed and then discovered post-hoc.
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
        """Map a dispatch exception to a disposition, preserving the prior classification."""
        # A ModelError from the client is already classified; re-raise it unchanged (the prior
        # `except ModelError: raise`), so it is not reclassified, retried, or its hold released.
        if isinstance(exc, ModelError):
            return DispatchOutcome(RetryDisposition.TERMINAL, lambda _n: exc)
        model_id = self._config.model_id
        if _is_timeout_error(exc):
            # Retryable, but capped at TIMEOUT_MAX_RETRIES (below the suite's max_retries) to bound
            # duplicate billing. On exhaustion it is stamped retryable=True, so the question becomes
            # auto-eligible for stage recovery on resume.
            return DispatchOutcome(
                RetryDisposition.RETRYABLE,
                lambda _n: ModelTimeoutError(
                    "Anthropic model request timed out.", model_id=model_id
                ),
                retry_cap=TIMEOUT_MAX_RETRIES,
            )
        if _is_authentication_error(exc):
            return DispatchOutcome(
                RetryDisposition.TERMINAL,
                lambda _n: ModelAuthenticationError(
                    "Anthropic authentication failed.", model_id=model_id
                ),
            )
        if _is_billing_error(exc):
            reason = _error_reason(exc)
            return DispatchOutcome(
                RetryDisposition.TERMINAL,
                lambda _n: ModelBillingError(
                    "Anthropic billing or quota is unavailable.", model_id=model_id, reason=reason
                ),
            )
        kind = _retryable_error_kind(exc)
        if kind == "rate_limit":
            return DispatchOutcome(
                RetryDisposition.RATE_LIMITED,
                lambda n: ModelRateLimitError(
                    "Anthropic rate limit persisted after retries.", model_id=model_id, attempts=n
                ),
                retry_after=retry_after_seconds(exc),
            )
        if kind is not None:
            return DispatchOutcome(
                RetryDisposition.RETRYABLE,
                lambda n: ModelError(
                    "Anthropic model request failed.", model_id=model_id, attempts=n
                ),
            )
        return DispatchOutcome(
            RetryDisposition.TERMINAL,
            lambda n: ModelError("Anthropic model request failed.", model_id=model_id, attempts=n),
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    def _require_client(self) -> _AnthropicClient:
        if self._client is None:
            raise ModelError("AnthropicModel is not initialized.")
        return self._client

    def _result_from_response(self, response: object, *, latency_ms: float) -> APICallResult:
        output = _extract_output_text(response)
        input_tokens = _extract_token_count(response, ("input_tokens",))
        output_tokens = _extract_token_count(response, ("output_tokens",))

        try:
            cost_usd = compute_cost_usd(
                self._config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=self._pricing,
            )
        except ValueError as exc:
            raise ModelError(
                "Anthropic pricing is not configured for the requested model.",
                model_id=self._config.model_id,
            ) from exc

        return APICallResult(
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model_id=self._config.model_id,
            raw_response=_compact_raw_response(response, output=output),
        )


def _create_anthropic_client(
    *,
    api_key: str,
    timeout_seconds: float,
) -> _AnthropicClient:
    from anthropic import AsyncAnthropic

    return cast(
        _AnthropicClient,
        AsyncAnthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0),
    )


def _extract_output_text(response: object) -> str:
    content = _field(response, "content")
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in cast(list[object], content):
            text = _field(block, "text")
            if isinstance(text, str):
                text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    raise ModelError("Anthropic response did not include output text.")


def _extract_token_count(response: object, names: tuple[str, ...]) -> int:
    usage = _field(response, "usage")
    for name in names:
        count = _field(usage, name)
        if isinstance(count, bool):
            continue
        if isinstance(count, int) and count >= 0:
            return count
    raise ModelError("Anthropic response did not include token usage.")


def _compact_raw_response(response: object, *, output: str) -> JsonObject:
    raw: JsonObject = {"output_text": output}
    for name in ("id", "model", "role", "stop_reason", "type"):
        value = _field(response, name)
        if _is_json_scalar(value):
            raw[name] = value

    usage = _field(response, "usage")
    usage_raw = _compact_usage(usage)
    if usage_raw:
        raw["usage"] = usage_raw
    return raw


def _compact_usage(usage: object) -> JsonObject:
    raw: JsonObject = {}
    for name in ("input_tokens", "output_tokens"):
        value = _field(usage, name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            raw[name] = value
    return raw


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
        "credit balance",
        "insufficient_quota",
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
