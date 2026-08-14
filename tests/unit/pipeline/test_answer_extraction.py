from __future__ import annotations

import pytest

from khedron.pipeline.generator import _extract_answer


def test_the_answer_is_the_text_after_the_marker() -> None:
    # The reference's prompt asks the model to reason in seven steps and commit after "ANSWER:".
    # Scoring the transcript instead of the commitment scores the reasoning, which is a different
    # and much more generous thing -- a rubric that accepts one correct item out of four will award
    # credit for facts mentioned on the way to a conclusion.
    output = "## Step 1: SCAN\nsome reasoning\n## Step 7\nANSWER: 2019"

    answer, failed = _extract_answer(output, "ANSWER:")

    if answer != "2019" or failed:
        raise AssertionError((answer, failed))


def test_the_last_marker_wins() -> None:
    # A reasoning transcript can quote its own instructions. rsplit, matching the reference's
    # `generated_answer.rsplit("ANSWER:", 1)[-1].strip()` exactly: a replication that extracts
    # differently is measuring a different string.
    output = "I must give my final answer after ANSWER: as instructed.\nANSWER: purple"

    answer, _failed = _extract_answer(output, "ANSWER:")

    if answer != "purple":
        raise AssertionError(answer)


def test_an_output_that_never_reached_the_marker_is_reported_as_a_failure() -> None:
    # A truncated transcript that never reaches the marker must be reported as a failure, not
    # scored as an answer. A low output-token ceiling against a verbose reasoning format can
    # truncate most answers mid-reasoning while still looking complete.
    truncated = "## Step 1: SCAN ALL MEMORIES\n1. (January 2022) Joanna congratulates Nate on"

    answer, failed = _extract_answer(truncated, "ANSWER:")

    if not failed:
        raise AssertionError("a truncated transcript was accepted as an answer")
    if answer != truncated:
        raise AssertionError(answer)


def test_a_profile_without_a_marker_passes_the_output_through() -> None:
    # canonical-v2's prompt asks for a bare answer, so there is nothing to extract and nothing that
    # can fail to arrive.
    answer, failed = _extract_answer("Blue", None)

    if answer != "Blue" or failed:
        raise AssertionError((answer, failed))


@pytest.mark.parametrize("profile_name", ["canonical-v1", "canonical-v2", "canonical-v2-baseline"])
def test_a_profile_declaring_no_marker_has_a_prompt_that_asks_for_none(profile_name: str) -> None:
    from khedron.methodology import get_runtime_profile

    profile = get_runtime_profile(profile_name)
    prompt = profile.generator_prompt_path.read_text(encoding="utf-8")

    if profile.answer_marker is not None:
        raise AssertionError(profile.answer_marker)
    if "ANSWER:" in prompt:
        raise AssertionError(
            f"{profile_name}'s prompt asks for a marker its profile does not declare"
        )
