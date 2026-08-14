from __future__ import annotations

import pytest
import tiktoken

from khedron.errors import KhedronError, ModelAuthenticationError, ModelBillingError
from khedron.utils import tokens
from khedron.utils.tokens import estimate_tokens


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeAnthropicMessages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None

    def count_tokens(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class _FakeAnthropicClient:
    def __init__(self, response: object) -> None:
        self.messages = _FakeAnthropicMessages(response)


class _FakeGoogleModels:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None

    def count_tokens(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class _FakeSDKError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeGoogleClient:
    def __init__(self, response: object) -> None:
        self.models = _FakeGoogleModels(response)


def test_openai_known_text_matches_cl100k_base() -> None:
    text = "hello world"
    expected = len(tiktoken.get_encoding("cl100k_base").encode(text))

    _check(estimate_tokens(text, "openai") == expected, "OpenAI token count should match")


def test_openai_empty_string_returns_zero() -> None:
    _check(estimate_tokens("", "openai") == 0, "empty text should have zero OpenAI tokens")


def test_anthropic_uses_default_model_and_message_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAnthropicClient({"input_tokens": 7})
    created_api_keys: list[str] = []

    def create_client(*, api_key: str) -> _FakeAnthropicClient:
        created_api_keys.append(api_key)
        return fake_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("KHEDRON_ANTHROPIC_TOKEN_MODEL", raising=False)
    monkeypatch.setattr(tokens, "_create_anthropic_client", create_client)

    count = estimate_tokens("hello", "anthropic")

    _check(count == 7, "Anthropic fake token count should pass through")
    _check(created_api_keys == ["test-anthropic-key"], "Anthropic API key should be used")
    _check(fake_client.messages.kwargs is not None, "Anthropic count_tokens should be called")
    _check(
        fake_client.messages.kwargs
        == {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "Anthropic count_tokens payload should match the contract",
    )


def test_anthropic_model_env_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAnthropicClient({"input_tokens": 9})

    def create_client(*, api_key: str) -> _FakeAnthropicClient:
        _ = api_key
        return fake_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("KHEDRON_ANTHROPIC_TOKEN_MODEL", "claude-custom")
    monkeypatch.setattr(tokens, "_create_anthropic_client", create_client)

    _check(estimate_tokens("hello", "anthropic") == 9, "Anthropic override should count")
    _check(fake_client.messages.kwargs is not None, "Anthropic count_tokens should be called")
    _check(
        fake_client.messages.kwargs["model"] == "claude-custom",
        "Anthropic env override should set the model",
    )


def test_anthropic_missing_api_key_raises_without_creating_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def create_client(*, api_key: str) -> _FakeAnthropicClient:
        nonlocal called
        called = True
        _ = api_key
        return _FakeAnthropicClient({"input_tokens": 1})

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(tokens, "_create_anthropic_client", create_client)

    with pytest.raises(ModelAuthenticationError):
        estimate_tokens("hello", "anthropic")

    _check(not called, "Anthropic client should not be created without an API key")


def test_google_uses_default_model_and_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeGoogleClient({"total_tokens": 11})
    created_api_keys: list[str] = []

    def create_client(*, api_key: str) -> _FakeGoogleClient:
        created_api_keys.append(api_key)
        return fake_client

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("KHEDRON_GOOGLE_TOKEN_MODEL", raising=False)
    monkeypatch.setattr(tokens, "_create_google_client", create_client)

    count = estimate_tokens("hello", "google")

    _check(count == 11, "Google fake token count should pass through")
    _check(created_api_keys == ["test-google-key"], "Google API key should be used")
    _check(fake_client.models.kwargs is not None, "Google count_tokens should be called")
    _check(
        fake_client.models.kwargs == {"model": "gemini-2.5-flash", "contents": "hello"},
        "Google count_tokens payload should match the contract",
    )


def test_google_model_env_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeGoogleClient({"totalTokens": 13})

    def create_client(*, api_key: str) -> _FakeGoogleClient:
        _ = api_key
        return fake_client

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("KHEDRON_GOOGLE_TOKEN_MODEL", "gemini-custom")
    monkeypatch.setattr(tokens, "_create_google_client", create_client)

    _check(estimate_tokens("hello", "google") == 13, "Google override should count")
    _check(fake_client.models.kwargs is not None, "Google count_tokens should be called")
    _check(
        fake_client.models.kwargs["model"] == "gemini-custom",
        "Google env override should set the model",
    )


def test_google_missing_api_key_raises_without_creating_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def create_client(*, api_key: str) -> _FakeGoogleClient:
        nonlocal called
        called = True
        _ = api_key
        return _FakeGoogleClient({"total_tokens": 1})

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(tokens, "_create_google_client", create_client)

    with pytest.raises(ModelAuthenticationError):
        estimate_tokens("hello", "google")

    _check(not called, "Google client should not be created without an API key")


@pytest.mark.parametrize(
    ("vendor", "response", "env_var", "factory_name"),
    [
        (
            "anthropic",
            _FakeSDKError(
                "Your credit balance is too low to access the Anthropic API.",
                status_code=400,
            ),
            "ANTHROPIC_API_KEY",
            "_create_anthropic_client",
        ),
        (
            "google",
            _FakeSDKError("RESOURCE_EXHAUSTED: quota exceeded.", status_code=429),
            "GOOGLE_API_KEY",
            "_create_google_client",
        ),
    ],
)
def test_billing_sdk_token_counter_errors_raise_framework_billing_error(
    monkeypatch: pytest.MonkeyPatch,
    vendor: str,
    response: object,
    env_var: str,
    factory_name: str,
) -> None:
    fake_client: object
    if vendor == "anthropic":
        fake_client = _FakeAnthropicClient(response)
    else:
        fake_client = _FakeGoogleClient(response)

    def create_client(*, api_key: str) -> object:
        _ = api_key
        return fake_client

    monkeypatch.setenv(env_var, "test-key")
    monkeypatch.setattr(tokens, factory_name, create_client)

    with pytest.raises(ModelBillingError):
        estimate_tokens("hello", vendor)  # type: ignore[arg-type]


def test_unsupported_runtime_vendor_raises_value_error() -> None:
    with pytest.raises(ValueError):
        estimate_tokens("hello", "not-a-vendor")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("vendor", "response", "env_var", "factory_name"),
    [
        ("anthropic", {}, "ANTHROPIC_API_KEY", "_create_anthropic_client"),
        ("google", {}, "GOOGLE_API_KEY", "_create_google_client"),
        ("anthropic", {"input_tokens": -1}, "ANTHROPIC_API_KEY", "_create_anthropic_client"),
        ("google", {"total_tokens": -1}, "GOOGLE_API_KEY", "_create_google_client"),
    ],
)
def test_missing_or_negative_sdk_token_counts_raise_framework_error(
    monkeypatch: pytest.MonkeyPatch,
    vendor: str,
    response: object,
    env_var: str,
    factory_name: str,
) -> None:
    fake_client: object
    if vendor == "anthropic":
        fake_client = _FakeAnthropicClient(response)
    else:
        fake_client = _FakeGoogleClient(response)

    def create_client(*, api_key: str) -> object:
        _ = api_key
        return fake_client

    monkeypatch.setenv(env_var, "test-key")
    monkeypatch.setattr(tokens, factory_name, create_client)

    with pytest.raises(KhedronError):
        estimate_tokens("hello", vendor)  # type: ignore[arg-type]
