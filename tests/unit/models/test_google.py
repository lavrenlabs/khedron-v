from __future__ import annotations

import pytest
from pydantic import ValidationError

from khedron.errors import (
    ModelAuthenticationError,
    ModelBillingError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from khedron.models import GoogleModel, GoogleModelConfig
from khedron.models.registry import get_model_class

MODEL_ID = "gemini-2.5-flash"


class FakeUsageMetadata:
    def __init__(
        self,
        *,
        prompt_token_count: int,
        candidates_token_count: int,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = prompt_token_count + candidates_token_count


class FakeResponse:
    def __init__(
        self,
        *,
        output_text: str = "Alice moved to Rome.",
        input_tokens: int = 100,
        output_tokens: int = 25,
    ) -> None:
        self.text = output_text
        self.model_version = MODEL_ID
        self.response_id = "resp-1"
        self.usage_metadata = FakeUsageMetadata(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
        )


class FakeGoogleError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeRateLimitError(FakeGoogleError):
    pass


class FakeServerError(FakeGoogleError):
    pass


class FakeAuthenticationError(FakeGoogleError):
    pass


class FakeBillingError(FakeGoogleError):
    pass


class FakeModelsAPI:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if not self._items:
            raise AssertionError("fake response queue is empty")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, items: list[object]) -> None:
        self.models = FakeModelsAPI(items)
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class FakeClientFactory:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> FakeClient:
        self.calls.append(
            {
                "api_key": api_key,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.client


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_google_config_defaults_and_validation() -> None:
    config = GoogleModelConfig(model_id=MODEL_ID)

    if config.model_id != MODEL_ID:
        raise AssertionError(config)
    if config.api_key_env_var != "GOOGLE_API_KEY":
        raise AssertionError(config)
    if config.timeout_seconds != 60.0:
        raise AssertionError(config)
    if config.max_retries != 3:
        raise AssertionError(config)

    with pytest.raises(ValidationError):
        GoogleModelConfig(model_id="")
    with pytest.raises(ValidationError):
        GoogleModelConfig(model_id=MODEL_ID, api_key_env_var="")
    with pytest.raises(ValidationError):
        GoogleModelConfig(model_id=MODEL_ID, timeout_seconds=0)
    with pytest.raises(ValidationError):
        GoogleModelConfig(model_id=MODEL_ID, max_retries=-1)


def test_google_model_is_registered() -> None:
    if get_model_class("google") is not GoogleModel:
        raise AssertionError("google model was not registered")


def test_model_properties_are_stable() -> None:
    model = GoogleModel(GoogleModelConfig(model_id=MODEL_ID))

    if model.model_id != MODEL_ID:
        raise AssertionError(model.model_id)
    if model.vendor != "google":
        raise AssertionError(model.vendor)


@pytest.mark.asyncio
async def test_missing_api_key_raises_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = FakeClient([FakeResponse()])
    factory = FakeClientFactory(client)
    model = GoogleModel(GoogleModelConfig(model_id=MODEL_ID), client_factory=factory)

    with pytest.raises(ModelAuthenticationError):
        await model.initialize()

    if factory.calls != []:
        raise AssertionError(factory.calls)


@pytest.mark.asyncio
async def test_initialize_creates_client_from_env_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_TEST_KEY", "test-key")
    client = FakeClient([FakeResponse()])
    factory = FakeClientFactory(client)
    model = GoogleModel(
        GoogleModelConfig(
            model_id=MODEL_ID,
            api_key_env_var="GOOGLE_TEST_KEY",
            timeout_seconds=12.5,
        ),
        client_factory=factory,
    )

    await model.initialize()
    await model.initialize()

    if factory.calls != [{"api_key": "test-key", "timeout_seconds": 12.5}]:
        raise AssertionError(factory.calls)


@pytest.mark.asyncio
async def test_generate_maps_success_response_and_computes_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient([FakeResponse(output_text="Rome", input_tokens=100, output_tokens=25)])
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    result = await model.generate("Where did Alice move?", max_output_tokens=64, temperature=0.2)

    if result.output != "Rome":
        raise AssertionError(result)
    if result.input_tokens != 100 or result.output_tokens != 25:
        raise AssertionError(result)
    if result.cost_usd != pytest.approx(0.0000925):
        raise AssertionError(result.cost_usd)
    if result.model_id != MODEL_ID:
        raise AssertionError(result)
    if result.latency_ms < 0:
        raise AssertionError(result)
    if result.raw_response["usage"] != {
        "prompt_token_count": 100,
        "candidates_token_count": 25,
        "total_token_count": 125,
    }:
        raise AssertionError(result.raw_response)
    if "api_key" in result.raw_response:
        raise AssertionError(result.raw_response)

    if client.models.calls != [
        {
            "model": MODEL_ID,
            "contents": "Where did Alice move?",
            "config": {
                "max_output_tokens": 64,
                "temperature": 0.2,
            },
        }
    ]:
        raise AssertionError(client.models.calls)


@pytest.mark.asyncio
async def test_retry_succeeds_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeRateLimitError("rate limited", status_code=429),
            FakeResponse(output_text="Recovered"),
        ]
    )
    sleep = SleepRecorder()
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    result = await model.generate("Question?")

    if result.output != "Recovered":
        raise AssertionError(result)
    if len(client.models.calls) != 2:
        raise AssertionError(client.models.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_rate_limit_exhaustion_raises_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeRateLimitError("rate limited", status_code=429),
            FakeRateLimitError("rate limited again", status_code=429),
        ]
    )
    sleep = SleepRecorder()
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelRateLimitError):
        await model.generate("Question?")

    if len(client.models.calls) != 2:
        raise AssertionError(client.models.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_billing_error_maps_to_framework_billing_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeBillingError(
                "RESOURCE_EXHAUSTED: quota exceeded for project.",
                status_code=429,
            )
        ]
    )
    sleep = SleepRecorder()
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID, max_retries=2),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelBillingError):
        await model.generate("Question?")

    if len(client.models.calls) != 1:
        raise AssertionError(client.models.calls)
    if sleep.delays != []:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_server_error_exhaustion_raises_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeServerError("server error", status_code=500),
            FakeServerError("server error again", status_code=503),
        ]
    )
    sleep = SleepRecorder()
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelError):
        await model.generate("Question?")

    if len(client.models.calls) != 2:
        raise AssertionError(client.models.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_authentication_error_maps_to_framework_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient([FakeAuthenticationError("bad key", status_code=401)])
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    with pytest.raises(ModelAuthenticationError):
        await model.generate("Question?")


@pytest.mark.asyncio
async def test_timeout_retries_then_exhausts_and_is_marked_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # A timeout is transient, so it now retries -- but capped at TIMEOUT_MAX_RETRIES (2), i.e. three
    # physical attempts, even with the larger max_retries (8) canonical suites use. On exhaustion
    # it is surfaced retryable=True, so the question is recoverable on resume.
    client = FakeClient([TimeoutError(f"deadline {i}") for i in range(5)])
    sleep = SleepRecorder()
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID, max_retries=8),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        await model.generate("Question?")

    if len(client.models.calls) != 3:
        raise AssertionError(client.models.calls)
    if sleep.delays != [1.0, 2.0]:
        raise AssertionError(sleep.delays)
    if exc_info.value.context.get("retryable") is not True:
        raise AssertionError(exc_info.value.context)


@pytest.mark.asyncio
async def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = FakeClient([FakeResponse()])
    model = GoogleModel(
        GoogleModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    await model.initialize()
    await model.close()
    await model.close()

    if client.close_count != 1:
        raise AssertionError(client.close_count)
