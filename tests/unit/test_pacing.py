from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from khedron.errors import ConfigurationError
from khedron.pacing import (
    RateLimitEntry,
    RateLimitsConfig,
    build_rate_limiter,
    estimate_reservation_tokens,
    retry_after_seconds,
)
from khedron.ratelimit import pacer_name


class _FakeHeaders:
    def __init__(self, value: object) -> None:
        self._value = value

    def get(self, name: str) -> object:
        return self._value if name == "retry-after" else None


class _FakeResponse:
    def __init__(self, value: object) -> None:
        self.headers = _FakeHeaders(value)


class _FakeRateLimitError(Exception):
    def __init__(self, retry_after: object) -> None:
        super().__init__("429")
        self.response = _FakeResponse(retry_after)


# --- build_rate_limiter ---


def test_no_config_or_empty_limits_builds_no_limiter() -> None:
    # None means "never enter the pacer", not "a pacer that admits everything" -- the two behave
    # identically only if we return None, so a run configuring nothing is exactly as before.
    if build_rate_limiter(None) is not None:
        raise AssertionError("None config must build no limiter")
    if build_rate_limiter(RateLimitsConfig()) is not None:
        raise AssertionError("empty limits must build no limiter")


def test_a_model_entry_builds_a_vendor_qualified_model_bucket() -> None:
    config = RateLimitsConfig(
        limits=(RateLimitEntry(vendor="openai", model="gpt-4o-mini", tokens_per_minute=200_000),)
    )
    limiter = build_rate_limiter(config)
    assert limiter is not None
    # The bucket exists under the vendor-qualified key and imposes the configured limit.
    bucket = limiter._path(pacer_name("openai", "gpt-4o-mini"), None)
    if not bucket or bucket[0].limit.tokens_per_minute != 200_000:
        raise AssertionError("model entry did not build the expected vendor-qualified bucket")


def test_the_same_scope_under_two_vendors_never_shares_a_bucket() -> None:
    # The whole point of vendor qualification: an "org" scope on OpenAI and on Anthropic are two
    # different accounts and must not be conflated into one limit.
    config = RateLimitsConfig(
        scope="org",
        limits=(
            RateLimitEntry(vendor="openai", tokens_per_minute=100),
            RateLimitEntry(vendor="anthropic", tokens_per_minute=100),
        ),
    )
    limiter = build_rate_limiter(config)
    assert limiter is not None
    openai_scope = limiter._path("x", pacer_name("openai", "org"))
    anthropic_scope = limiter._path("y", pacer_name("anthropic", "org"))
    if openai_scope[0] is anthropic_scope[0]:
        raise AssertionError("two vendors' scopes collapsed into one bucket")


def test_a_scope_entry_without_a_configured_scope_is_rejected() -> None:
    config = RateLimitsConfig(limits=(RateLimitEntry(vendor="openai", tokens_per_minute=100),))
    with pytest.raises(ConfigurationError):
        build_rate_limiter(config)


def test_two_entries_resolving_to_one_bucket_are_rejected() -> None:
    config = RateLimitsConfig(
        limits=(
            RateLimitEntry(vendor="openai", model="m", tokens_per_minute=100),
            RateLimitEntry(vendor="openai", model="m", tokens_per_minute=200),
        )
    )
    with pytest.raises(ConfigurationError):
        build_rate_limiter(config)


# --- estimate_reservation_tokens ---


class _FakeEncoding:
    """A stand-in tiktoken encoding: one token per character, so the count is predictable."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


def test_the_openai_estimate_uses_the_models_own_encoding_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Offline: the real o200k encoding downloads from a URL on a cold cache, so a unit test must not
    # depend on it. Mock encoding_for_model, assert it is called with the model id (binding "the
    # model's own encoding"), and assert get_encoding("cl100k_base") is never used as a fallback.
    calls: list[str] = []

    def fake_encoding_for_model(model_id: str) -> _FakeEncoding:
        calls.append(model_id)
        return _FakeEncoding()

    def forbidden_get_encoding(name: str) -> _FakeEncoding:
        raise AssertionError(
            f"the generic get_encoding({name!r}) must not be used for a known model"
        )

    monkeypatch.setattr("khedron.pacing.tiktoken.encoding_for_model", fake_encoding_for_model)
    monkeypatch.setattr("khedron.pacing.tiktoken.get_encoding", forbidden_get_encoding)

    prompt = "abcdefghij"  # 10 chars -> 10 fake tokens
    got = estimate_reservation_tokens(
        vendor="openai", model_id="gpt-4o-mini", prompt=prompt, max_output_tokens=256
    )
    if calls != ["gpt-4o-mini"]:
        raise AssertionError(f"encoding_for_model was not called with the model id: {calls}")
    if got != 10 + 256 + 32:
        raise AssertionError(f"openai estimate {got} != tokens + max_output + overhead")


def test_the_openai_estimate_falls_back_to_o200k_for_an_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare/future model id makes encoding_for_model raise KeyError; the estimator must fall back
    # to o200k_base, not cl100k. Both mocked so the test stays offline.
    used: list[str] = []

    def raising_encoding_for_model(model_id: str) -> _FakeEncoding:
        raise KeyError(model_id)

    def fake_get_encoding(name: str) -> _FakeEncoding:
        used.append(name)
        return _FakeEncoding()

    monkeypatch.setattr("khedron.pacing.tiktoken.encoding_for_model", raising_encoding_for_model)
    monkeypatch.setattr("khedron.pacing.tiktoken.get_encoding", fake_get_encoding)

    estimate_reservation_tokens(
        vendor="openai", model_id="some-future-model", prompt="abc", max_output_tokens=0
    )
    if used != ["o200k_base"]:
        raise AssertionError(f"the fallback must be o200k_base, got {used}")


def test_the_non_openai_estimate_is_a_byte_upper_bound() -> None:
    prompt = "grade this answer"
    got = estimate_reservation_tokens(
        vendor="anthropic", model_id="claude", prompt=prompt, max_output_tokens=80
    )
    if got != len(prompt.encode("utf-8")) + 80 + 32:
        raise AssertionError(got)


def test_a_multibyte_prompt_is_not_underestimated_the_way_chars_over_four_would() -> None:
    # The adversarial case: CJK and emoji. chars/4 would badly underestimate the token count on
    # multibyte text; the byte upper bound cannot. (That bytes >= tokens for a byte-level BPE
    # tokeniser is a property of the tokeniser, not something to re-verify with a network download.)
    prompt = "日本語のテキストです。" * 10 + "🎉🔥🚀" * 10
    naive_chars_over_four = len(prompt) // 4
    anthropic_body = (
        estimate_reservation_tokens(
            vendor="anthropic", model_id="claude", prompt=prompt, max_output_tokens=0
        )
        - 32
    )
    if anthropic_body <= naive_chars_over_four:
        raise AssertionError("the byte bound must exceed a chars/4 estimate on multibyte text")
    # A 3-byte CJK glyph and a 4-byte emoji both count for more than one, so the byte bound is
    # strictly larger than the character count -- the naive under-count the guarantee rules out.
    if anthropic_body <= len(prompt):
        raise AssertionError("the byte bound must exceed the character count on multibyte text")


# --- retry_after_seconds ---


def test_retry_after_reads_delta_seconds() -> None:
    if retry_after_seconds(_FakeRateLimitError("3")) != pytest.approx(3.0):
        raise AssertionError("a delta-seconds Retry-After must be read")


def test_retry_after_reads_an_http_date() -> None:
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    exc = _FakeRateLimitError("Thu, 01 Jan 2026 00:00:10 GMT")
    if retry_after_seconds(exc, now=now) != pytest.approx(10.0):
        raise AssertionError("an HTTP-date Retry-After must be read as a delta")


def test_retry_after_treats_a_past_date_as_zero() -> None:
    now = datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
    exc = _FakeRateLimitError("Thu, 01 Jan 2026 00:00:10 GMT")
    if retry_after_seconds(exc, now=now) != pytest.approx(0.0):
        raise AssertionError("a Retry-After in the past must be zero, not negative")


def test_retry_after_is_zero_when_absent_or_malformed() -> None:
    if retry_after_seconds(_FakeRateLimitError(None)) != 0.0:
        raise AssertionError("absent Retry-After must be 0.0")
    if retry_after_seconds(_FakeRateLimitError("not-a-date")) != 0.0:
        raise AssertionError("malformed Retry-After must be 0.0")
    if retry_after_seconds(_FakeRateLimitError("-5")) != 0.0:
        raise AssertionError("negative delta-seconds must be 0.0")
    if retry_after_seconds(Exception("no response attribute")) != 0.0:
        raise AssertionError("an exception with no response must yield 0.0")


# --- config validation: empty-limit entries and empty/invalid names ---


def test_an_entry_with_no_token_or_request_limit_is_rejected() -> None:
    # It constrains nothing, so building a limiter from it contradicts "None when unconfigured".
    for entry in (
        RateLimitEntry(vendor="openai", model="m"),
        RateLimitEntry(vendor="openai"),  # a scope entry, still limitless
    ):
        scope = None if entry.model is not None else "org"
        with pytest.raises(ConfigurationError):
            build_rate_limiter(RateLimitsConfig(scope=scope, limits=(entry,)))


def test_empty_scope_or_model_names_are_rejected_at_config_time() -> None:
    with pytest.raises(ValidationError):
        RateLimitEntry(vendor="openai", model="", tokens_per_minute=100)
    with pytest.raises(ValidationError):
        RateLimitsConfig(scope="", limits=())


def test_a_control_character_in_a_name_becomes_a_configuration_error() -> None:
    # pacer_name raises ValueError on the qualifier control char; from config that must surface as a
    # ConfigurationError, not a bare ValueError.
    config = RateLimitsConfig(
        limits=(RateLimitEntry(vendor="open\x1fai", model="m", tokens_per_minute=100),)
    )
    with pytest.raises(ConfigurationError):
        build_rate_limiter(config)


# --- retry_after defensiveness ---


def test_retry_after_reads_the_capitalised_header_spelling() -> None:
    # A plain dict is case-sensitive; the SDK's headers object is not. Trying both spellings covers
    # a test double or atypical exception that only carries the capitalised form.
    class _CapitalisedHeaders:
        def get(self, name: str) -> object:
            return "7" if name == "Retry-After" else None

    class _Exc(Exception):
        response = type("R", (), {"headers": _CapitalisedHeaders()})()

    if retry_after_seconds(_Exc()) != pytest.approx(7.0):
        raise AssertionError("the capitalised Retry-After spelling must be read")


def test_retry_after_swallows_a_raising_header_accessor() -> None:
    class _RaisingHeaders:
        def get(self, name: str) -> object:
            raise RuntimeError("header access blew up")

    class _Exc(Exception):
        response = type("R", (), {"headers": _RaisingHeaders()})()

    if retry_after_seconds(_Exc()) != 0.0:
        raise AssertionError("a header accessor that raises must yield 0.0, not propagate")


def test_retry_after_swallows_a_raising_response_property() -> None:
    class _Exc(Exception):
        @property
        def response(self) -> object:
            raise RuntimeError("response property blew up")

    if retry_after_seconds(_Exc()) != 0.0:
        raise AssertionError("a response property that raises must yield 0.0, not propagate")


# --- the loader is strict: unknown keys are errors, not silently dropped ---


def test_rate_limits_models_forbid_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        RateLimitsConfig.model_validate({"limits": [], "typo": 1})
    with pytest.raises(ValidationError):
        RateLimitEntry.model_validate({"vendor": "openai", "tokens_per_minute": 100, "typpo": 1})
    # Nested: an unknown key inside a limits entry must also be rejected.
    with pytest.raises(ValidationError):
        RateLimitsConfig.model_validate(
            {"limits": [{"vendor": "openai", "tokens_per_minute": 100, "oops": 1}]}
        )


def test_load_rate_limits_config_reads_a_valid_file(tmp_path: object) -> None:
    from pathlib import Path

    from khedron.pacing import load_rate_limits_config

    path = Path(str(tmp_path)) / "rl.yaml"
    path.write_text(
        "scope: proj\nlimits:\n  - vendor: openai\n    model: m\n    tokens_per_minute: 100\n",
        encoding="utf-8",
    )
    config = load_rate_limits_config(path)
    if config.scope != "proj" or config.limits[0].tokens_per_minute != 100:
        raise AssertionError(config)


def test_load_rate_limits_config_rejects_a_typo_as_configuration_error(tmp_path: object) -> None:
    from pathlib import Path

    from khedron.pacing import load_rate_limits_config

    path = Path(str(tmp_path)) / "rl.yaml"
    path.write_text("limits: []\ntypo: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_rate_limits_config(path)
