from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from khedron.publishability import resolve_disposition
from khedron.types import RunStartedEvent

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _run(
    *,
    profile: str = "canonical-v2",
    framework_version: str = "0.1.0+abc1234",
    runtime_environment: dict[str, Any] | None = None,
) -> RunStartedEvent:
    return RunStartedEvent(
        event_id="e",
        timestamp=NOW,
        run_id="r",
        sequence_number=0,
        suite_id="s",
        experiment_id="x",
        experiment_name="X",
        run_number=0,
        provider_type="full_context",
        provider_version="1",
        benchmark_type="locomo",
        benchmark_version="1.0",
        benchmark_checksum="sha256:x",
        answer_model_id="gpt-4o-mini-2024-07-18",
        answer_model_vendor="openai",
        judge_model_id="claude-haiku-4-5-20251001",
        judge_model_vendor="anthropic",
        config={},
        methodology_version="2.0",
        methodology_profile=profile,
        framework_version=framework_version,
        seed=1,
        # , not : an empty dict is falsy, and falling back to the full-coverage
        # default silently turned the unrecorded-coverage case into the opposite test.
        runtime_environment=(
            {"corpus_conversations_evaluated": 10, "corpus_conversations_available": 10}
            if runtime_environment is None
            else runtime_environment
        ),
    )


def test_a_full_corpus_run_under_a_current_profile_is_publishable() -> None:
    disposition = resolve_disposition(_run())

    if not disposition.publishable or disposition.reasons:
        raise AssertionError(disposition)


def test_a_superseded_profile_withdraws_without_anyone_recording_a_decision() -> None:
    # The day-1 results are the case this exists for: measured under a profile that pinned no
    # model, no retrieval budget and no category set. They resolve as withdrawn because of what
    # their own record says, not because someone remembered to withdraw them.
    disposition = resolve_disposition(_run(profile="canonical-v1"))

    if disposition.publishable:
        raise AssertionError(disposition)
    joined = " ".join(disposition.reasons)
    if "canonical-v1" not in joined or "superseded" not in joined:
        raise AssertionError(disposition.reasons)
    if "pins neither" not in joined:
        raise AssertionError("an unpinned profile must also be named as a reason")


def test_a_partial_corpus_withdraws() -> None:
    disposition = resolve_disposition(
        _run(
            runtime_environment={
                "corpus_conversations_evaluated": 1,
                "corpus_conversations_available": 10,
            }
        )
    )

    if disposition.publishable:
        raise AssertionError(disposition)
    if "1 of 10" not in " ".join(disposition.reasons):
        raise AssertionError(disposition.reasons)


def test_an_unidentifiable_build_withdraws() -> None:
    for version in ("0.1.0+abc1234.dirty", "0.0.0+unidentified"):
        disposition = resolve_disposition(_run(framework_version=version))
        if disposition.publishable:
            raise AssertionError((version, disposition))


def test_a_rejudged_run_is_not_an_independent_measurement() -> None:
    # It re-scores another run's answers. Publishing it as a measurement would double-count the
    # generation it did not perform.
    disposition = resolve_disposition(
        _run(
            runtime_environment={
                "corpus_conversations_evaluated": 10,
                "corpus_conversations_available": 10,
                "answers_regenerated": False,
            }
        )
    )

    if disposition.publishable:
        raise AssertionError(disposition)
    if "re-scored" not in " ".join(disposition.reasons):
        raise AssertionError(disposition.reasons)


def test_an_unresolvable_profile_fails_closed() -> None:
    # A run whose disposition cannot be established is not publishable. A false withdrawal costs a
    # number nobody publishes; a false publication is what this project exists to prevent.
    disposition = resolve_disposition(_run(profile="a-profile-that-does-not-exist"))

    if disposition.publishable:
        raise AssertionError(disposition)
    if "cannot be resolved" not in " ".join(disposition.reasons):
        raise AssertionError(disposition.reasons)


def test_unrecorded_corpus_coverage_withdraws_rather_than_passing() -> None:
    # The resolver promised to fail closed and did the opposite for exactly this field: the check
    # fired only when both counts were present integers and one was smaller, so absent, non-numeric
    # or zero coverage produced no reason at all and the run resolved as publishable.
    for environment in (
        {},
        {"corpus_conversations_evaluated": "all", "corpus_conversations_available": 10},
        {"corpus_conversations_evaluated": 0, "corpus_conversations_available": 0},
        {"corpus_conversations_evaluated": True, "corpus_conversations_available": True},
        {"corpus_conversations_evaluated": 10},
    ):
        disposition = resolve_disposition(_run(runtime_environment=environment))
        if disposition.publishable:
            raise AssertionError(f"unknown coverage resolved as publishable: {environment}")
