from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, cast

from khedron.utils.redaction import looks_like_credential

__all__ = [
    "BenchmarkChecksumError",
    "BenchmarkError",
    "BenchmarkLoadError",
    "ConfigurationError",
    "CostExceededError",
    "JudgeError",
    "JudgeMalformedResponseError",
    "KhedronError",
    "ModelAuthenticationError",
    "ModelBillingError",
    "ModelError",
    "ModelRateLimitError",
    "ModelTimeoutError",
    "PersistenceError",
    "ProviderError",
    "ProviderInitializationError",
    "ProviderProtocolError",
    "ProviderTimeoutError",
    "SchemaVersionError",
]

_REDACTED: Final = "<redacted>"
_SENSITIVE_KEY_PARTS: Final = (
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)


class KhedronError(Exception):
    """Base exception for all Khedron framework errors."""

    context: dict[str, Any]

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context = deepcopy(context)

    @property
    def redacted_context(self) -> dict[str, Any]:
        """The context with credentials removed, for anything that persists it structurally.

        ``__str__`` was the only redacted view, so a consumer that copied ``context`` itself --
        ``ErrorRecord.context``, written to ``errors.jsonl`` -- persisted the raw values while
        the message beside it was clean. One definition, so the two views cannot drift apart.
        """
        return {
            key: _REDACTED if _is_sensitive_key(key) else _redact_nested_context(value)
            for key, value in self.context.items()
        }

    def __str__(self) -> str:
        message = super().__str__()
        if not self.context:
            return message

        context_items = ", ".join(
            f"{key}={self._stringify_context_value(key, value)}"
            for key, value in sorted(self.context.items())
        )
        return f"{message} ({context_items})"

    @staticmethod
    def _stringify_context_value(key: str, value: Any) -> str:
        if _is_sensitive_key(key):
            return _REDACTED
        return repr(_redact_nested_context(value))


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_nested_context(value: Any) -> Any:
    # A credential-shaped value is redacted whatever key it sits under. The key-name rule
    # above cannot carry this on its own: `str(exc)` is persisted as `error_message` and
    # projected into SQLite, and a provider that reports a failure with
    # `binary_path=<the value it was handed>` leaks whatever the operator put in the plugin
    # config block -- which is the one place a credential legitimately lives. Key names in
    # diagnostic context are arbitrary, so unlike configuration this cannot be an allowlist:
    # an allowlist here would erase the diagnostics the message exists for.
    if isinstance(value, str) and looks_like_credential(value):
        return _REDACTED
    if isinstance(value, dict):
        value_dict = cast(dict[Any, Any], value)
        return {
            key: _REDACTED if _is_sensitive_key(str(key)) else _redact_nested_context(nested_value)
            for key, nested_value in value_dict.items()
        }
    if isinstance(value, list):
        value_list = cast(list[Any], value)
        return [_redact_nested_context(item) for item in value_list]
    if isinstance(value, tuple):
        value_tuple = cast(tuple[Any, ...], value)
        return tuple(_redact_nested_context(item) for item in value_tuple)
    if isinstance(value, set):
        value_set = cast(set[Any], value)
        return {_redact_nested_context(item) for item in value_set}
    return value


class ConfigurationError(KhedronError):
    """Raised when benchmark configuration is invalid or incomplete."""


class ProviderError(KhedronError):
    """Raised when a memory provider operation fails."""


class ProviderInitializationError(ProviderError):
    """Raised when a memory provider cannot initialize successfully."""


class ProviderTimeoutError(ProviderError):
    """Raised when a memory provider operation exceeds its timeout."""


class ProviderProtocolError(ProviderError):
    """Raised when a memory provider returns an invalid protocol response."""


class ModelError(KhedronError):
    """Raised when an answer model operation fails."""


class ModelRateLimitError(ModelError):
    """Raised when an answer model request is rate limited."""


class ModelAuthenticationError(ModelError):
    """Raised when an answer model rejects configured authentication."""


class ModelBillingError(ModelError):
    """Raised when a model provider rejects a request for billing or quota reasons."""


class ModelTimeoutError(ModelError):
    """Raised when an answer model request exceeds its timeout."""


class JudgeError(KhedronError):
    """Raised when a judge operation fails."""


class JudgeMalformedResponseError(JudgeError):
    """Raised when a judge returns a response that cannot be parsed."""


class BenchmarkError(KhedronError):
    """Raised when benchmark loading or execution fails."""


class BenchmarkLoadError(BenchmarkError):
    """Raised when benchmark data cannot be loaded."""


class BenchmarkChecksumError(BenchmarkError):
    """Raised when benchmark data does not match its expected checksum."""


class PersistenceError(KhedronError):
    """Raised when benchmark persistence fails."""


class SchemaVersionError(PersistenceError):
    """Raised when persisted data has an unsupported schema version."""


class CostExceededError(KhedronError):
    """Raised when configured benchmark cost limits are exceeded."""
