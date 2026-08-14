from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, Protocol, cast

import tiktoken

from khedron.errors import (
    ConfigurationError,
    ModelAuthenticationError,
    ModelBillingError,
    ModelError,
)

_OPENAI_ENCODING = "cl100k_base"
_DEFAULT_ANTHROPIC_TOKEN_MODEL = "claude-sonnet-4-5"  # noqa: S105
_DEFAULT_GOOGLE_TOKEN_MODEL = "gemini-2.5-flash"  # noqa: S105
_ANTHROPIC_TOKEN_MODEL_ENV_VAR = "KHEDRON_ANTHROPIC_TOKEN_MODEL"  # noqa: S105
_GOOGLE_TOKEN_MODEL_ENV_VAR = "KHEDRON_GOOGLE_TOKEN_MODEL"  # noqa: S105


class _AnthropicMessagesAPI(Protocol):
    def count_tokens(self, **kwargs: object) -> object: ...


class _AnthropicClient(Protocol):
    messages: _AnthropicMessagesAPI


class _GoogleModelsAPI(Protocol):
    def count_tokens(self, **kwargs: object) -> object: ...


class _GoogleClient(Protocol):
    models: _GoogleModelsAPI


def estimate_tokens(text: str, vendor: Literal["openai", "anthropic", "google"]) -> int:
    """Estimate token count for a string using the vendor's tokenization scheme."""
    if vendor == "openai":
        return _estimate_openai_tokens(text)
    if vendor == "anthropic":
        return _estimate_anthropic_tokens(text)
    if vendor == "google":
        return _estimate_google_tokens(text)
    raise ValueError(f"Unsupported token estimation vendor: {vendor}")


def _estimate_openai_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding(_OPENAI_ENCODING)
    return len(encoding.encode(text))


def _estimate_anthropic_tokens(text: str) -> int:
    api_key = _require_api_key("ANTHROPIC_API_KEY", vendor="Anthropic")
    model = _env_or_default(_ANTHROPIC_TOKEN_MODEL_ENV_VAR, _DEFAULT_ANTHROPIC_TOKEN_MODEL)
    client = _create_anthropic_client(api_key=api_key)
    try:
        response = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
    except Exception as exc:
        raise _map_token_counter_error(exc, vendor="Anthropic") from exc
    return _extract_token_count(
        response,
        names=("input_tokens", "inputTokens", "tokens", "token_count", "tokenCount"),
        vendor="Anthropic",
    )


def _estimate_google_tokens(text: str) -> int:
    api_key = _require_api_key("GOOGLE_API_KEY", vendor="Google")
    model = _env_or_default(_GOOGLE_TOKEN_MODEL_ENV_VAR, _DEFAULT_GOOGLE_TOKEN_MODEL)
    client = _create_google_client(api_key=api_key)
    try:
        response = client.models.count_tokens(model=model, contents=text)
    except Exception as exc:
        raise _map_token_counter_error(exc, vendor="Google") from exc
    return _extract_token_count(
        response,
        names=("total_tokens", "totalTokens", "total_token_count", "totalTokenCount"),
        vendor="Google",
    )


def _create_anthropic_client(*, api_key: str) -> _AnthropicClient:
    from anthropic import Anthropic

    return cast(_AnthropicClient, Anthropic(api_key=api_key))


def _create_google_client(*, api_key: str) -> _GoogleClient:
    from google import genai

    return cast(_GoogleClient, genai.Client(api_key=api_key))


def _require_api_key(env_var: str, *, vendor: str) -> str:
    api_key = os.environ.get(env_var)
    if api_key is None or not api_key.strip():
        raise ModelAuthenticationError(
            f"{vendor} API key is not configured for token estimation.",
            api_key_env_var=env_var,
        )
    return api_key


def _env_or_default(env_var: str, default: str) -> str:
    value = os.environ.get(env_var)
    if value is None or not value.strip():
        return default
    return value


def _extract_token_count(response: object, *, names: tuple[str, ...], vendor: str) -> int:
    for name in names:
        count = _field(response, name)
        if isinstance(count, bool):
            continue
        if isinstance(count, int) and count >= 0:
            return count
        if isinstance(count, int) and count < 0:
            raise ConfigurationError(
                f"{vendor} token counter returned a negative token count.",
                token_count=count,
            )
    raise ModelError(f"{vendor} token counter did not return a token count.")


def _map_token_counter_error(exc: Exception, *, vendor: str) -> ModelError:
    if _is_authentication_error(exc):
        return ModelAuthenticationError(
            f"{vendor} token counter authentication failed.",
            vendor=vendor,
        )
    if _is_billing_error(exc):
        return ModelBillingError(
            f"{vendor} token counter billing or quota is unavailable.",
            vendor=vendor,
            reason=_error_reason(exc),
        )
    return ModelError(f"{vendor} token counter failed.", vendor=vendor)


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


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, bool):
        return None
    if isinstance(status_code, int):
        return status_code

    code = getattr(exc, "code", None)
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code
    return None


def _field(source: object, name: str) -> object:
    if source is None:
        return None
    if isinstance(source, Mapping):
        value = cast(Mapping[str, object], source).get(name)
        return value
    return getattr(source, name, None)
