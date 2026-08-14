from __future__ import annotations

import pytest
from pydantic import ValidationError

from khedron.budget import BudgetLedger, BudgetReservationError
from khedron.errors import (
    ModelAuthenticationError,
    ModelBillingError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from khedron.models import OpenAIModel, OpenAIModelConfig
from khedron.models.registry import get_model_class

MODEL_ID = "gpt-4o-mini-2024-07-18"


class _FakeTokenDetails:
    def __init__(self, *, cached_tokens: int) -> None:
        self.cached_tokens = cached_tokens


class FakeUsage:
    def __init__(
        self, *, input_tokens: int, output_tokens: int, cached_tokens: int | None = None
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        # Present only on a cache hit, mirroring the Responses API's usage.input_tokens_details.
        if cached_tokens is not None:
            self.input_tokens_details = _FakeTokenDetails(cached_tokens=cached_tokens)


class FakeResponse:
    def __init__(
        self,
        *,
        output_text: str = "Alice moved to Rome.",
        input_tokens: int = 100,
        output_tokens: int = 25,
        cached_tokens: int | None = None,
    ) -> None:
        self.id = "resp-1"
        self.model = MODEL_ID
        self.output_text = output_text
        self.usage = FakeUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens
        )


class FakeOpenAIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class FakeRateLimitError(FakeOpenAIError):
    pass


class FakeServerError(FakeOpenAIError):
    pass


class FakeAuthenticationError(FakeOpenAIError):
    pass


class FakeBillingError(FakeOpenAIError):
    pass


class FakeConnectionError(FakeOpenAIError):
    """Mirrors the SDK's APIConnectionError: a transient failure with no status code."""


class FakeResponsesAPI:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if not self._items:
            raise AssertionError("fake response queue is empty")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, items: list[object]) -> None:
        self.responses = FakeResponsesAPI(items)
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


class FakeClientFactory:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        timeout_seconds: float,
    ) -> FakeClient:
        self.calls.append(
            {
                "api_key": api_key,
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.client


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_openai_config_defaults_and_validation() -> None:
    config = OpenAIModelConfig(model_id=MODEL_ID)

    if config.model_id != MODEL_ID:
        raise AssertionError(config)
    if config.api_key_env_var != "OPENAI_API_KEY":
        raise AssertionError(config)
    if config.base_url is not None:
        raise AssertionError(config)
    if config.timeout_seconds != 60.0:
        raise AssertionError(config)
    if config.max_retries != 3:
        raise AssertionError(config)

    with pytest.raises(ValidationError):
        OpenAIModelConfig(model_id="")
    with pytest.raises(ValidationError):
        OpenAIModelConfig(model_id=MODEL_ID, api_key_env_var="")
    with pytest.raises(ValidationError):
        OpenAIModelConfig(model_id=MODEL_ID, timeout_seconds=0)
    with pytest.raises(ValidationError):
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=-1)


def test_openai_model_is_registered() -> None:
    if get_model_class("openai") is not OpenAIModel:
        raise AssertionError("openai model was not registered")


def test_model_properties_are_stable() -> None:
    model = OpenAIModel(OpenAIModelConfig(model_id=MODEL_ID))

    if model.model_id != MODEL_ID:
        raise AssertionError(model.model_id)
    if model.vendor != "openai":
        raise AssertionError(model.vendor)


@pytest.mark.asyncio
async def test_missing_api_key_raises_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FakeClient([FakeResponse()])
    factory = FakeClientFactory(client)
    model = OpenAIModel(OpenAIModelConfig(model_id=MODEL_ID), client_factory=factory)

    with pytest.raises(ModelAuthenticationError):
        await model.initialize()

    if factory.calls != []:
        raise AssertionError(factory.calls)


@pytest.mark.asyncio
async def test_initialize_creates_client_from_env_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    client = FakeClient([FakeResponse()])
    factory = FakeClientFactory(client)
    model = OpenAIModel(
        OpenAIModelConfig(
            model_id=MODEL_ID,
            api_key_env_var="OPENAI_TEST_KEY",
            base_url="https://example.invalid/v1",
            timeout_seconds=12.5,
        ),
        client_factory=factory,
    )

    await model.initialize()
    await model.initialize()

    if factory.calls != [
        {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "timeout_seconds": 12.5,
        }
    ]:
        raise AssertionError(factory.calls)


@pytest.mark.asyncio
async def test_generate_maps_success_response_and_computes_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([FakeResponse(output_text="Rome", input_tokens=100, output_tokens=25)])
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    result = await model.generate("Where did Alice move?", max_output_tokens=64, temperature=0.2)

    if result.output != "Rome":
        raise AssertionError(result)
    if result.input_tokens != 100 or result.output_tokens != 25:
        raise AssertionError(result)
    if result.cost_usd != pytest.approx(0.00003):
        raise AssertionError(result.cost_usd)
    if result.model_id != MODEL_ID:
        raise AssertionError(result)
    if result.latency_ms < 0:
        raise AssertionError(result)
    if result.raw_response["usage"] != {
        "input_tokens": 100,
        "output_tokens": 25,
        "total_tokens": 125,
    }:
        raise AssertionError(result.raw_response)
    # A response with no cache-hit details reports zero cached tokens and adds no usage key.
    if result.cached_input_tokens != 0:
        raise AssertionError(result.cached_input_tokens)
    if "api_key" in result.raw_response:
        raise AssertionError(result.raw_response)

    if client.responses.calls != [
        {
            "model": MODEL_ID,
            "input": [{"role": "user", "content": "Where did Alice move?"}],
            "max_output_tokens": 64,
            "temperature": 0.2,
        }
    ]:
        raise AssertionError(client.responses.calls)


@pytest.mark.asyncio
async def test_generate_captures_cached_input_tokens_when_the_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache hit's cached_tokens is captured on the result and mirrored into the raw usage.

    Cached tokens are a subset of input_tokens; they are recorded for observability but do not
    reduce cost (input_tokens is still billed in full), so cost_usd matches the no-cache case.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [FakeResponse(output_text="Rome", input_tokens=100, output_tokens=25, cached_tokens=64)]
    )
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    result = await model.generate("Where did Alice move?", max_output_tokens=64, temperature=0.0)

    if result.cached_input_tokens != 64:
        raise AssertionError(result.cached_input_tokens)
    if result.input_tokens != 100:
        raise AssertionError(result.input_tokens)
    if result.cost_usd != pytest.approx(0.00003):
        raise AssertionError(result.cost_usd)
    if result.raw_response["usage"].get("cached_input_tokens") != 64:
        raise AssertionError(result.raw_response["usage"])


@pytest.mark.asyncio
async def test_generate_drops_a_cached_count_that_exceeds_input_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cached_tokens is a subset of input; a count above input is dropped, not persisted."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [FakeResponse(output_text="Rome", input_tokens=100, output_tokens=25, cached_tokens=101)]
    )
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    result = await model.generate("Where did Alice move?", max_output_tokens=64, temperature=0.0)

    if result.cached_input_tokens != 0:
        raise AssertionError(result.cached_input_tokens)
    if "cached_input_tokens" in result.raw_response["usage"]:
        raise AssertionError(result.raw_response["usage"])


@pytest.mark.asyncio
async def test_retry_succeeds_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeRateLimitError("rate limited", status_code=429),
            FakeResponse(output_text="Recovered"),
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    result = await model.generate("Question?")

    if result.output != "Recovered":
        raise AssertionError(result)
    if len(client.responses.calls) != 2:
        raise AssertionError(client.responses.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_rate_limit_exhaustion_raises_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeRateLimitError("rate limited", status_code=429),
            FakeRateLimitError("rate limited again", status_code=429),
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelRateLimitError):
        await model.generate("Question?")

    if len(client.responses.calls) != 2:
        raise AssertionError(client.responses.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_without_a_rate_limiter_the_token_estimate_is_never_computed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The unpaced path must do no pacer work: no reservation, and no token estimate -- computing it
    # loads a tiktoken encoding (a cache/temp dir, possibly a download). tiktoken is imported at
    # module load either way; what must not happen without a limiter is the encoding load, which
    # only the estimate triggers. A run with no rate limits behaves exactly as before. Bound by
    # making the estimate blow up if reached.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def exploding_estimate(**_kwargs: object) -> int:
        raise AssertionError("the token estimate must not run without a rate limiter")

    monkeypatch.setattr("khedron.models.openai.estimate_reservation_tokens", exploding_estimate)
    client = FakeClient([FakeResponse(output_text="Rome")])
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    result = await model.generate("Where did Alice move?")

    if result.output != "Rome":
        raise AssertionError(result)


@pytest.mark.asyncio
async def test_a_model_error_from_the_client_passes_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Preserves the prior `except ModelError: raise`. A ModelError raised by the client is already
    # classified; the pacer must not reclassify it as a fresh rate limit, retry it, or release its
    # hold. Asserted as identity and zero retries.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sentinel = ModelRateLimitError("already classified", model_id=MODEL_ID, attempts=1)
    client = FakeClient([sentinel])
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=3),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelRateLimitError) as exc_info:
        await model.generate("Question?")

    if exc_info.value is not sentinel:
        raise AssertionError("a ModelError from the client must pass through as the same instance")
    if len(client.responses.calls) != 1:
        raise AssertionError(
            f"a passed-through ModelError must not be retried: {client.responses.calls}"
        )
    if sleep.delays:
        raise AssertionError(f"a passed-through ModelError must not back off: {sleep.delays}")


@pytest.mark.asyncio
async def test_billing_error_maps_to_framework_billing_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeBillingError(
                "You exceeded your current quota.",
                status_code=429,
                code="insufficient_quota",
            )
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=2),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelBillingError):
        await model.generate("Question?")

    if len(client.responses.calls) != 1:
        raise AssertionError(client.responses.calls)
    if sleep.delays != []:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_server_error_exhaustion_raises_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeServerError("server error", status_code=500),
            FakeServerError("server error again", status_code=503),
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelError):
        await model.generate("Question?")

    if len(client.responses.calls) != 2:
        raise AssertionError(client.responses.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_retry_succeeds_after_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeConnectionError("Connection error."),
            FakeResponse(output_text="Recovered"),
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    result = await model.generate("Question?")

    if result.output != "Recovered":
        raise AssertionError(result)
    if len(client.responses.calls) != 2:
        raise AssertionError(client.responses.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_connection_error_exhaustion_raises_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient(
        [
            FakeConnectionError("Connection error."),
            FakeConnectionError("Connection error again."),
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=1),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelError):
        await model.generate("Question?")

    if len(client.responses.calls) != 2:
        raise AssertionError(client.responses.calls)
    # Without a limiter the backoff is the unchanged fixed exponential: one 1.0s wait before the
    # single retry. The pacer's full jitter applies only when rate limits are configured.
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_authentication_error_maps_to_framework_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([FakeAuthenticationError("bad key", status_code=401)])
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    with pytest.raises(ModelAuthenticationError):
        await model.generate("Question?")


@pytest.mark.asyncio
async def test_timeout_retries_then_exhausts_and_is_marked_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # A timeout is transient, so it now retries -- but capped at TIMEOUT_MAX_RETRIES (2), i.e. three
    # physical attempts, even though this config sets a larger max_retries (8, as canonical suites
    # do). Only three of the five queued timeouts are consumed, bounding duplicate billing. On
    # exhaustion the error is surfaced retryable=True, which makes the question auto-eligible for
    # stage recovery on the next resume.
    client = FakeClient([TimeoutError(f"deadline {i}") for i in range(5)])
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=8),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        await model.generate("Question?")

    if len(client.responses.calls) != 3:
        raise AssertionError(client.responses.calls)
    if sleep.delays != [1.0, 2.0]:
        raise AssertionError(sleep.delays)
    if exc_info.value.context.get("retryable") is not True:
        raise AssertionError(exc_info.value.context)


@pytest.mark.asyncio
async def test_timeout_then_server_errors_stay_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # A timeout latches the retry ceiling down for the whole call: after a timeout (possibly already
    # billed), later server errors cannot push the call past TIMEOUT_MAX_RETRIES. With max_retries=8
    # and a timeout first, exactly three physical attempts occur -- not nine.
    client = FakeClient(
        [
            TimeoutError("deadline exceeded"),
            FakeServerError("server error", status_code=500),
            FakeServerError("server error again", status_code=503),
            FakeServerError("still failing", status_code=500),
            FakeServerError("still failing", status_code=500),
        ]
    )
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=8),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelError):
        await model.generate("Question?")

    if len(client.responses.calls) != 3:
        raise AssertionError(client.responses.calls)
    if sleep.delays != [1.0, 2.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_server_errors_are_not_capped_by_the_timeout_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # The cap is timeout-specific. A call that never times out is governed entirely by max_retries:
    # six server errors under max_retries=5 make the full six physical attempts, not three.
    client = FakeClient([FakeServerError(f"error {i}", status_code=500) for i in range(6)])
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID, max_retries=5),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    with pytest.raises(ModelError):
        await model.generate("Question?")

    if len(client.responses.calls) != 6:
        raise AssertionError(client.responses.calls)
    if sleep.delays != [1.0, 2.0, 4.0, 8.0, 16.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_timeout_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([TimeoutError("deadline exceeded"), FakeResponse(output_text="Recovered")])
    sleep = SleepRecorder()
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
        backoff_sleep=sleep,
    )

    result = await model.generate("Question?")

    if result.output != "Recovered":
        raise AssertionError(result)
    if len(client.responses.calls) != 2:
        raise AssertionError(client.responses.calls)
    if sleep.delays != [1.0]:
        raise AssertionError(sleep.delays)


@pytest.mark.asyncio
async def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([FakeResponse()])
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
    )

    await model.initialize()
    await model.close()
    await model.close()

    if client.close_count != 1:
        raise AssertionError(client.close_count)


# --- The money reservation on the real adapter through paced_generate ---


@pytest.mark.asyncio
async def test_an_unpriceable_model_refuses_before_dispatch_on_the_programmatic_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the real Runner(...).run() code path (this adapter's generate(), not CLI
    # preflight): a model absent from the pricing table cannot be bounded, so it fails closed BEFORE
    # the paid call is dispatched -- the fake client's create() is never reached, nothing is billed.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([FakeResponse()])
    model = OpenAIModel(
        OpenAIModelConfig(model_id="unpriced-model-zzz"),
        client_factory=FakeClientFactory(client),
        budget=BudgetLedger(cap_provider=lambda: 100.0),
    )

    with pytest.raises(ModelError):
        await model.generate("hello", max_output_tokens=64)

    if client.responses.calls:
        raise AssertionError("an unpriceable model must not dispatch a paid call")


@pytest.mark.asyncio
async def test_a_call_that_would_breach_the_cap_is_refused_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bound on the real adapter: with a zero cap, this call's worst-case cost is over budget, so
    # the reservation refuses before dispatch and the paid call is never sent.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([FakeResponse()])
    ledger = BudgetLedger(cap_provider=lambda: 0.0)
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
        budget=ledger,
    )

    with pytest.raises(BudgetReservationError):
        await model.generate("hello", max_output_tokens=64)

    if client.responses.calls:
        raise AssertionError("a call over the cap must not be dispatched")
    if ledger.worst_case_cost_usd() != pytest.approx(0.0):
        raise AssertionError("a refused call must leave no hold behind")


@pytest.mark.asyncio
async def test_a_priced_call_under_the_cap_dispatches_and_settles_to_observed_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The happy path with a budget: the call dispatches, and settle replaces the reserved maximum
    # with the real billed cost, so the ledger's observed equals the result cost and there is no
    # residual (worst_case == observed).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeClient([FakeResponse(input_tokens=100, output_tokens=25)])
    ledger = BudgetLedger(cap_provider=lambda: 100.0)
    model = OpenAIModel(
        OpenAIModelConfig(model_id=MODEL_ID),
        client_factory=FakeClientFactory(client),
        budget=ledger,
    )

    result = await model.generate("Where did Alice move?", max_output_tokens=64)

    if len(client.responses.calls) != 1:
        raise AssertionError("a call under the cap must be dispatched exactly once")
    if ledger.observed_cost_usd() != pytest.approx(result.cost_usd):
        raise AssertionError("settle must record the real billed cost as observed spend")
    if ledger.worst_case_cost_usd() != pytest.approx(ledger.observed_cost_usd()):
        raise AssertionError("a fully-settled run leaves no ambiguous residual")
