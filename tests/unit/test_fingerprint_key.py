from __future__ import annotations

from datetime import UTC, datetime

from khedron.methodology import (
    METHODOLOGY_FINGERPRINT_KEY,
    get_runtime_profile,
    methodology_fingerprint,
)
from khedron.rejudge import _derived_run_started
from khedron.types import RunStartedEvent


class _Judge:
    model_id = "claude-haiku-4-5-20251001"
    vendor = "anthropic"


def _source(fingerprint: str) -> RunStartedEvent:
    return RunStartedEvent(
        event_id="e",
        timestamp=datetime(2026, 8, 1, tzinfo=UTC),
        run_id="source",
        sequence_number=0,
        suite_id="s",
        experiment_id="x",
        experiment_name="X",
        run_number=0,
        provider_type="full_context",
        provider_version="1",
        benchmark_type="locomo",
        benchmark_version="1",
        benchmark_checksum="sha256:x",
        answer_model_id="gpt-4o-mini-2024-07-18",
        answer_model_vendor="openai",
        judge_model_id="gpt-4o",
        judge_model_vendor="openai",
        config={},
        methodology_version="2.0",
        methodology_profile="canonical-v2",
        framework_version="0.1.0+abc1234",
        seed=1,
        runtime_environment={METHODOLOGY_FINGERPRINT_KEY: fingerprint},
    )


def test_a_derived_run_discloses_its_own_fingerprint_not_the_one_it_inherited() -> None:
    # Three spellings of one key meant the derived run kept the source's value under the name the
    # report reads, and wrote its own under a name nothing read. Three recorded runs behind the
    # dossier disclose a fingerprint that is not theirs because of it. Asserted with a source value
    # that is deliberately not any real fingerprint, so inheriting it cannot pass by coincidence.
    profile_name = "canonical-v2-baseline"
    derived = _derived_run_started(
        run_id="derived",
        sequence=0,
        profile=get_runtime_profile(profile_name),
        source=_source("inherited-and-wrong"),
        judge=_Judge(),
    )

    recorded = derived.runtime_environment[METHODOLOGY_FINGERPRINT_KEY]
    if recorded == "inherited-and-wrong":
        raise AssertionError("the derived run kept the fingerprint it inherited from its source")
    if recorded != methodology_fingerprint(get_runtime_profile(profile_name)):
        raise AssertionError(recorded)


def test_no_second_spelling_of_the_key_survives_anywhere() -> None:
    # The defect was two readers and two writers disagreeing on a string. Guarded rather than
    # remembered: a reintroduced literal is invisible until a guard that should fire does not.
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[2] / "src" / "khedron"
    offenders = [
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        # The two shapes that matter -- reading it and writing it. The bare name also appears as an
        # exported function in __all__, which is not a key and must not be flagged.
        if any(
            spelling in path.read_text(encoding="utf-8")
            for spelling in ('get("methodology_fingerprint")', '"methodology_fingerprint":')
        )
    ]
    if offenders:
        raise AssertionError(f"a second spelling of the fingerprint key survives in: {offenders}")
