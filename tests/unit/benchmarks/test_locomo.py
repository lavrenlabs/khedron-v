from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from khedron.benchmarks import LocomoBenchmark
from khedron.benchmarks.locomo import EXPECTED_DATASET_SHA256
from khedron.benchmarks.registry import get_benchmark_class
from khedron.errors import BenchmarkChecksumError, BenchmarkLoadError
from khedron.types import QuestionCategory

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DATASET_PATH = PROJECT_ROOT / "data" / "locomo" / "locomo10.json"
CANONICAL_AUDIT_PATH = PROJECT_ROOT / "data" / "audits" / "locomo_errors.json"

# The LoCoMo corpus (CC BY-NC 4.0, Snap Research) is not distributed with the framework; fetch it
# with the download script. Tests that exercise the real corpus skip when it is absent, so offline
# CI stays green without redistributing a non-commercial dataset. Tests that build their own tiny
# in-memory fixtures do not depend on it and are unaffected.
pytestmark = pytest.mark.skipif(
    not CANONICAL_DATASET_PATH.exists(),
    reason="LoCoMo dataset absent; run the download script under scripts/ to fetch it.",
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> str:
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_empty_audit(path: Path) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "source": {"included_count": 0},
            "errors": [],
        },
    )


def minimal_dataset(category: int = 4, answer: object = "answer") -> list[dict[str, object]]:
    question: dict[str, object] = {
        "question": "What did Alice say?",
        "answer": answer,
        "evidence": ["D1:1"],
        "category": category,
    }
    return [
        {
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {
                        "speaker": "Alice",
                        "dia_id": "D1:1",
                        "text": "Hello Bob.",
                        "img_url": "https://example.invalid/image.jpg",
                    }
                ],
            },
            "qa": [question],
        }
    ]


@pytest.mark.asyncio
async def test_locomo_properties_registration_and_idempotent_load() -> None:
    benchmark = LocomoBenchmark()

    if benchmark.benchmark_type != "locomo":
        raise AssertionError(benchmark.benchmark_type)
    if benchmark.benchmark_version != "1.0":
        raise AssertionError(benchmark.benchmark_version)
    if benchmark.dataset_checksum != EXPECTED_DATASET_SHA256:
        raise AssertionError(benchmark.dataset_checksum)
    if get_benchmark_class("locomo") is not LocomoBenchmark:
        raise AssertionError(get_benchmark_class("locomo"))

    await benchmark.load()
    first_conversations = benchmark.get_conversations()
    first_questions = benchmark.get_questions()
    await benchmark.load()

    if benchmark.get_conversations() != first_conversations:
        raise AssertionError(benchmark.get_conversations())
    if benchmark.get_questions() != first_questions:
        raise AssertionError(benchmark.get_questions())


@pytest.mark.asyncio
async def test_canonical_dataset_counts_and_category_distribution() -> None:
    benchmark = LocomoBenchmark()

    await benchmark.load()
    conversations = benchmark.get_conversations()
    questions = benchmark.get_questions()
    counts = Counter(question.category.value for question in questions)
    answerable_subset = [
        question for question in questions if question.category is not QuestionCategory.ADVERSARIAL
    ]

    if len(conversations) != 10:
        raise AssertionError(len(conversations))
    if len(questions) != 1986:
        raise AssertionError(len(questions))
    if len(answerable_subset) != 1540:
        raise AssertionError(len(answerable_subset))
    if counts != {
        "multi_hop": 282,
        "temporal": 321,
        "open_domain": 96,
        "single_hop": 841,
        "adversarial": 446,
    }:
        raise AssertionError(counts)

    first_conversation = conversations[0]
    if first_conversation.conversation_id != "conv-26":
        raise AssertionError(first_conversation.conversation_id)
    if first_conversation.speakers != ["Caroline", "Melanie"]:
        raise AssertionError(first_conversation.speakers)
    if first_conversation.sessions[0].session_id != "conv-26_session_1":
        raise AssertionError(first_conversation.sessions[0])
    if first_conversation.sessions[0].timestamp.tzinfo is None:
        raise AssertionError(first_conversation.sessions[0].timestamp)


@pytest.mark.asyncio
async def test_get_questions_filters_by_conversation_and_category() -> None:
    benchmark = LocomoBenchmark()
    await benchmark.load()

    all_questions = benchmark.get_questions()
    conversation_id = benchmark.get_conversations()[0].conversation_id
    expected_conversation_questions = [
        question for question in all_questions if question.conversation_id == conversation_id
    ]
    expected_single_hop = [
        question for question in all_questions if question.category is QuestionCategory.SINGLE_HOP
    ]
    expected_combined = [
        question
        for question in expected_conversation_questions
        if question.category is QuestionCategory.SINGLE_HOP
    ]

    if benchmark.get_questions(conversation_id=conversation_id) != expected_conversation_questions:
        raise AssertionError(benchmark.get_questions(conversation_id=conversation_id))
    if benchmark.get_questions(categories=["single_hop"]) != expected_single_hop:
        raise AssertionError(benchmark.get_questions(categories=["single_hop"]))
    if (
        benchmark.get_questions(conversation_id=conversation_id, categories=["single_hop"])
        != expected_combined
    ):
        raise AssertionError(
            benchmark.get_questions(conversation_id=conversation_id, categories=["single_hop"])
        )


@pytest.mark.asyncio
async def test_configured_conversation_and_category_filters_select_quickstart_subset() -> None:
    conversation_only = LocomoBenchmark(conversations=["conv-30"])
    quickstart_subset = LocomoBenchmark(
        conversations=["conv-30"],
        categories=["multi_hop"],
    )

    await conversation_only.load()
    await quickstart_subset.load()

    conversations = quickstart_subset.get_conversations()
    questions = quickstart_subset.get_questions()

    if len(conversation_only.get_questions()) != 105:
        raise AssertionError(len(conversation_only.get_questions()))
    if [conversation.conversation_id for conversation in conversations] != ["conv-30"]:
        raise AssertionError(conversations)
    if len(questions) != 11:
        raise AssertionError(len(questions))
    if {question.conversation_id for question in questions} != {"conv-30"}:
        raise AssertionError(questions)
    if {question.category.value for question in questions} != {"multi_hop"}:
        raise AssertionError(questions)
    if quickstart_subset.get_questions(conversation_id="conv-30") != questions:
        raise AssertionError(quickstart_subset.get_questions(conversation_id="conv-30"))
    if quickstart_subset.get_questions(categories=["single_hop"]):
        raise AssertionError(quickstart_subset.get_questions(categories=["single_hop"]))


@pytest.mark.asyncio
async def test_configured_filter_validation_fails_clearly() -> None:
    with pytest.raises(BenchmarkLoadError, match="list of strings"):
        LocomoBenchmark(conversations=cast(list[str], "conv-30"))

    with pytest.raises(BenchmarkLoadError, match="unknown category"):
        LocomoBenchmark(categories=["unknown_category"])

    with pytest.raises(BenchmarkLoadError, match="unknown conversation IDs"):
        await LocomoBenchmark(conversations=["missing-conversation"]).load()

    with pytest.raises(BenchmarkLoadError, match="selected no questions"):
        await LocomoBenchmark(categories=[]).load()


@pytest.mark.asyncio
async def test_audit_artifact_marks_exactly_compiled_score_corrupting_errors() -> None:
    benchmark = LocomoBenchmark()
    await benchmark.load()

    audit_artifact = json.loads(CANONICAL_AUDIT_PATH.read_text(encoding="utf-8"))
    artifact_errors = audit_artifact["errors"]
    artifact_question_ids = {error["question_id"] for error in artifact_errors}
    marked_question_ids = {
        question.question_id for question in benchmark.get_questions() if question.is_audited_error
    }

    if len(benchmark.get_audit_errors()) != 99:
        raise AssertionError(len(benchmark.get_audit_errors()))
    if benchmark.get_audit_errors() != artifact_question_ids:
        raise AssertionError(benchmark.get_audit_errors())
    if marked_question_ids != artifact_question_ids:
        raise AssertionError(marked_question_ids)
    if any(error["error_type"] == "WRONG_CITATION" for error in artifact_errors):
        raise AssertionError(artifact_errors)


@pytest.mark.asyncio
async def test_category_5_null_answers_are_preserved_with_d0013_metadata() -> None:
    benchmark = LocomoBenchmark()
    await benchmark.load()

    null_answer_questions = [
        question
        for question in benchmark.get_questions(categories=["adversarial"])
        if question.metadata["raw_answer_is_null"] is True
    ]

    if len(null_answer_questions) != 444:
        raise AssertionError(len(null_answer_questions))
    if any(question.expected_answer != "" for question in null_answer_questions):
        raise AssertionError(null_answer_questions[:3])
    if any(question.metadata["answerable"] is not False for question in null_answer_questions):
        raise AssertionError(null_answer_questions[:3])
    if any(question.metadata["raw_answer"] is not None for question in null_answer_questions):
        raise AssertionError(null_answer_questions[:3])


@pytest.mark.asyncio
async def test_checksum_mismatch_raises_checksum_error(tmp_path: Path) -> None:
    copied_dataset = tmp_path / "locomo10.json"
    copied_dataset.write_bytes(CANONICAL_DATASET_PATH.read_bytes() + b"\n")
    benchmark = LocomoBenchmark(
        dataset_path=copied_dataset,
        audit_path=CANONICAL_AUDIT_PATH,
    )

    with pytest.raises(BenchmarkChecksumError):
        await benchmark.load()


@pytest.mark.asyncio
async def test_missing_dataset_or_audit_artifact_raises_load_error(tmp_path: Path) -> None:
    missing_dataset = tmp_path / "missing-locomo10.json"
    missing_audit = tmp_path / "missing-audit.json"

    with pytest.raises(BenchmarkLoadError):
        await LocomoBenchmark(dataset_path=missing_dataset).load()

    with pytest.raises(BenchmarkLoadError):
        await LocomoBenchmark(
            dataset_path=CANONICAL_DATASET_PATH,
            audit_path=missing_audit,
            expected_dataset_checksum=file_digest(CANONICAL_DATASET_PATH),
        ).load()


@pytest.mark.asyncio
async def test_malformed_dataset_raises_load_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo10.json"
    audit_path = tmp_path / "locomo_errors.json"
    malformed_payload = b"{"
    dataset_path.write_bytes(malformed_payload)
    write_empty_audit(audit_path)

    benchmark = LocomoBenchmark(
        dataset_path=dataset_path,
        audit_path=audit_path,
        expected_dataset_checksum=hashlib.sha256(malformed_payload).hexdigest(),
    )

    with pytest.raises(BenchmarkLoadError, match="malformed JSON"):
        await benchmark.load()


@pytest.mark.asyncio
async def test_unknown_category_entry_raises_load_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo10.json"
    audit_path = tmp_path / "locomo_errors.json"
    expected_checksum = write_json(dataset_path, minimal_dataset(category=99))
    write_empty_audit(audit_path)

    benchmark = LocomoBenchmark(
        dataset_path=dataset_path,
        audit_path=audit_path,
        expected_dataset_checksum=expected_checksum,
    )

    with pytest.raises(BenchmarkLoadError, match="unknown category"):
        await benchmark.load()


@pytest.mark.asyncio
async def test_invalid_audit_artifact_raises_load_error(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo10.json"
    audit_path = tmp_path / "locomo_errors.json"
    expected_checksum = write_json(dataset_path, minimal_dataset())
    write_json(
        audit_path,
        {
            "schema_version": 1,
            "source": {"included_count": 1},
            "errors": [
                {
                    "question_id": "locomo_0_qa0",
                    "error_type": "WRONG_CITATION",
                }
            ],
        },
    )

    benchmark = LocomoBenchmark(
        dataset_path=dataset_path,
        audit_path=audit_path,
        expected_dataset_checksum=expected_checksum,
    )

    with pytest.raises(BenchmarkLoadError, match="WRONG_CITATION"):
        await benchmark.load()


def test_image_descriptions_are_ingested_only_when_the_profile_asks() -> None:
    # 1,226 of LoCoMo's 5,882 turns carry a description of an image the speaker shared. Khedron
    # parsed it into metadata and ingested only the spoken text, so a fifth of the corpus never
    # reached the memory store and questions about what a photo showed were unanswerable by
    # construction. Off by default because turning it on changes what was measured, not how well.
    import asyncio

    def turns(include: bool) -> list[str]:
        benchmark = LocomoBenchmark(conversations=["conv-44"], include_image_descriptions=include)
        asyncio.run(benchmark.load())
        return [
            turn.content
            for conversation in benchmark.get_conversations()
            for session in conversation.sessions
            for turn in session.turns
        ]

    without, with_images = turns(False), turns(True)

    if len(without) != len(with_images):
        raise AssertionError("the turn count must not change; only their text does")
    if any("shared an image depicting:" in text for text in without):
        raise AssertionError("descriptions leaked in with the flag off")
    described = [text for text in with_images if "shared an image depicting:" in text]
    if len(described) < 100:
        raise AssertionError(f"expected many described turns in conv-44, got {len(described)}")
    # Khedron's own rendering choice: the caption only. The dataset's `query` -- what its authors
    # searched to find the image -- contradicts the caption systematically and never enters memory.
    if not described[0].endswith("]") or "query" in described[0]:
        raise AssertionError(described[0][-120:])
    if sum(map(len, with_images)) <= sum(map(len, without)):
        raise AssertionError("no content was actually added")


def test_a_turn_without_an_image_is_untouched_when_descriptions_are_on() -> None:
    import asyncio

    benchmark = LocomoBenchmark(conversations=["conv-44"], include_image_descriptions=True)
    asyncio.run(benchmark.load())
    plain = [
        turn.content
        for conversation in benchmark.get_conversations()
        for session in conversation.sessions
        for turn in session.turns
        if not turn.metadata.get("blip_caption")
    ]

    if not plain:
        raise AssertionError("conv-44 should contain turns with no image at all")
    if any("shared an image depicting:" in text for text in plain):
        raise AssertionError("a turn with no caption was given one")
