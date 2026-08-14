from __future__ import annotations

from typing import Any

import pytest

from khedron.errors import (
    BenchmarkChecksumError,
    BenchmarkError,
    BenchmarkLoadError,
    ConfigurationError,
    CostExceededError,
    JudgeError,
    JudgeMalformedResponseError,
    KhedronError,
    ModelAuthenticationError,
    ModelBillingError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
    PersistenceError,
    ProviderError,
    ProviderInitializationError,
    ProviderProtocolError,
    ProviderTimeoutError,
    SchemaVersionError,
)

EXCEPTION_CLASSES = (
    KhedronError,
    ConfigurationError,
    ProviderError,
    ProviderInitializationError,
    ProviderTimeoutError,
    ProviderProtocolError,
    ModelError,
    ModelRateLimitError,
    ModelAuthenticationError,
    ModelBillingError,
    ModelTimeoutError,
    JudgeError,
    JudgeMalformedResponseError,
    BenchmarkError,
    BenchmarkLoadError,
    BenchmarkChecksumError,
    PersistenceError,
    SchemaVersionError,
    CostExceededError,
)


@pytest.mark.parametrize(
    ("exception_class", "expected_parent"),
    [
        (KhedronError, Exception),
        (ConfigurationError, KhedronError),
        (ProviderError, KhedronError),
        (ProviderInitializationError, ProviderError),
        (ProviderTimeoutError, ProviderError),
        (ProviderProtocolError, ProviderError),
        (ModelError, KhedronError),
        (ModelRateLimitError, ModelError),
        (ModelAuthenticationError, ModelError),
        (ModelBillingError, ModelError),
        (ModelTimeoutError, ModelError),
        (JudgeError, KhedronError),
        (JudgeMalformedResponseError, JudgeError),
        (BenchmarkError, KhedronError),
        (BenchmarkLoadError, BenchmarkError),
        (BenchmarkChecksumError, BenchmarkError),
        (PersistenceError, KhedronError),
        (SchemaVersionError, PersistenceError),
        (CostExceededError, KhedronError),
    ],
)
def test_exception_hierarchy(
    exception_class: type[Exception],
    expected_parent: type[Exception],
) -> None:
    if not issubclass(exception_class, expected_parent):
        raise AssertionError((exception_class, expected_parent))
    if exception_class is not KhedronError:
        if not issubclass(exception_class, KhedronError):
            raise AssertionError(exception_class)


@pytest.mark.parametrize("exception_class", EXCEPTION_CLASSES)
def test_exception_classes_store_message_and_context(
    exception_class: type[KhedronError],
) -> None:
    error = exception_class("operation failed", provider="full_context", attempt=2)

    if error.args != ("operation failed",):
        raise AssertionError(error.args)
    if error.context != {"provider": "full_context", "attempt": 2}:
        raise AssertionError(error.context)
    if str(error) != "operation failed (attempt=2, provider='full_context')":
        raise AssertionError(str(error))


def test_context_is_isolated_from_caller_mutation() -> None:
    mutable_context: dict[str, Any] = {"phase": "retrieve", "attempts": [1, 2]}

    error = ProviderError("provider failed", details=mutable_context)
    mutable_context["phase"] = "generate"
    mutable_context["attempts"].append(3)

    if error.context != {"details": {"phase": "retrieve", "attempts": [1, 2]}}:
        raise AssertionError(error.context)


def test_string_representation_redacts_sensitive_context_values() -> None:
    sensitive_value = "-".join(["do", "not", "show"])

    error = ModelAuthenticationError(
        "authentication failed",
        api_key=sensitive_value,
        provider="openai",
    )

    rendered = str(error)

    if rendered != "authentication failed (api_key=<redacted>, provider='openai')":
        raise AssertionError(rendered)
    if sensitive_value in rendered:
        raise AssertionError(rendered)
    if error.context["api_key"] != sensitive_value:
        raise AssertionError(error.context)


def test_string_representation_recursively_redacts_sensitive_context_values() -> None:
    api_key = "-".join(["nested", "api", "key"])
    auth_value = " ".join(["Bearer", "-".join(["nested", "auth"])])
    token_value = "-".join(["nested", "token"])
    secret_value = "-".join(["nested", "secret"])

    error = ProviderProtocolError(
        "provider returned invalid response",
        headers={"Authorization": auth_value, "Content-Type": "application/json"},
        metadata={
            "api_key": api_key,
            "attempt": 3,
            "events": [
                {"name": "request_started", "token": token_value},
                ("status", {"secret": secret_value}),
            ],
        },
    )

    rendered = str(error)

    if "Content-Type" not in rendered or "application/json" not in rendered:
        raise AssertionError(rendered)
    if "attempt" not in rendered or "3" not in rendered:
        raise AssertionError(rendered)
    if "request_started" not in rendered:
        raise AssertionError(rendered)
    for sensitive_value in (api_key, auth_value, token_value, secret_value):
        if sensitive_value in rendered:
            raise AssertionError(rendered)
    if rendered.count("<redacted>") != 4:
        raise AssertionError(rendered)
    if error.context["headers"]["Authorization"] != auth_value:
        raise AssertionError(error.context)
    if error.context["metadata"]["api_key"] != api_key:
        raise AssertionError(error.context)


@pytest.mark.parametrize("exception_class", EXCEPTION_CLASSES)
def test_public_exception_classes_have_docstrings(
    exception_class: type[KhedronError],
) -> None:
    if exception_class.__doc__ is None:
        raise AssertionError(exception_class)
    if not exception_class.__doc__.strip():
        raise AssertionError(exception_class)
