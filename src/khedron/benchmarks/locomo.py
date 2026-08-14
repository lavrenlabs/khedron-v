from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from khedron.benchmarks.base import Benchmark
from khedron.benchmarks.registry import register_benchmark
from khedron.errors import BenchmarkChecksumError, BenchmarkLoadError
from khedron.types import Conversation, Question, QuestionCategory, Session, Turn

__all__ = ["EXPECTED_DATASET_SHA256", "LocomoBenchmark"]

EXPECTED_DATASET_SHA256: Final[str] = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
BENCHMARK_VERSION: Final[str] = "1.0"
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET_PATH: Final[Path] = _PROJECT_ROOT / "data" / "locomo" / "locomo10.json"
_DEFAULT_AUDIT_PATH: Final[Path] = _PROJECT_ROOT / "data" / "audits" / "locomo_errors.json"
_SESSION_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^session_(\d+)$")
_OPTIONAL_TURN_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "img_url",
    "blip_caption",
    "query",
    "re-download",
)
_CATEGORY_MAP: Final[dict[int, QuestionCategory]] = {
    1: QuestionCategory.MULTI_HOP,
    2: QuestionCategory.TEMPORAL,
    3: QuestionCategory.OPEN_DOMAIN,
    4: QuestionCategory.SINGLE_HOP,
    5: QuestionCategory.ADVERSARIAL,
}


@register_benchmark("locomo")
class LocomoBenchmark(Benchmark):
    """Loader for the canonical LoCoMo 10-conversation benchmark dataset."""

    def __init__(
        self,
        dataset_path: Path | None = None,
        audit_path: Path | None = None,
        expected_dataset_checksum: str = EXPECTED_DATASET_SHA256,
        conversations: list[str] | None = None,
        categories: list[str] | None = None,
        include_image_descriptions: bool = False,
    ) -> None:
        self._dataset_path = dataset_path if dataset_path is not None else _DEFAULT_DATASET_PATH
        self._audit_path = audit_path if audit_path is not None else _DEFAULT_AUDIT_PATH
        self._expected_dataset_checksum = expected_dataset_checksum.lower()
        self._conversation_filter = _optional_str_filter(conversations, "conversations")
        self._category_filter = _optional_category_filter(categories)
        self._include_image_descriptions = include_image_descriptions
        self._conversations: list[Conversation] = []
        self._questions: list[Question] = []
        self._audit_errors: set[str] = set()
        self._corpus_conversation_count = 0
        self._loaded = False

    @property
    def benchmark_type(self) -> str:
        return "locomo"

    @property
    def benchmark_version(self) -> str:
        return BENCHMARK_VERSION

    @property
    def dataset_checksum(self) -> str:
        return self._expected_dataset_checksum

    async def load(self) -> None:
        if self._loaded:
            return

        conversations, questions, audit_errors = await asyncio.to_thread(self._load_sync)
        self._conversations = conversations
        self._questions = questions
        self._audit_errors = audit_errors
        self._loaded = True

    def get_conversations(self) -> list[Conversation]:
        return list(self._conversations)

    def get_questions(
        self,
        conversation_id: str | None = None,
        categories: list[str] | None = None,
    ) -> list[Question]:
        questions = list(self._questions)
        if conversation_id is not None:
            questions = [
                question for question in questions if question.conversation_id == conversation_id
            ]
        if categories is not None:
            category_values = set(categories)
            questions = [
                question for question in questions if question.category.value in category_values
            ]
        return questions

    @property
    def corpus_conversation_count(self) -> int:
        """Conversations in the dataset before `conversations:` narrowed it."""
        return self._corpus_conversation_count

    def get_audit_errors(self) -> set[str]:
        return set(self._audit_errors)

    def _load_sync(self) -> tuple[list[Conversation], list[Question], set[str]]:
        actual_checksum = _sha256_file(self._dataset_path)
        if actual_checksum != self._expected_dataset_checksum:
            raise BenchmarkChecksumError(
                "LoCoMo dataset checksum mismatch.",
                path=str(self._dataset_path),
                expected=self._expected_dataset_checksum,
                observed=actual_checksum,
            )

        dataset = _load_json(self._dataset_path, artifact_name="LoCoMo dataset")
        audit_errors = _load_audit_errors(self._audit_path)
        conversations, questions = _parse_dataset(
            dataset, audit_errors, include_image_descriptions=self._include_image_descriptions
        )
        question_ids = {question.question_id for question in questions}
        unknown_audit_ids = audit_errors - question_ids
        if unknown_audit_ids:
            raise BenchmarkLoadError(
                "LoCoMo audit artifact references unknown question IDs.",
                path=str(self._audit_path),
                unknown_question_ids=sorted(unknown_audit_ids),
            )
        self._corpus_conversation_count = len(conversations)
        conversations, questions = _apply_configured_filters(
            conversations,
            questions,
            conversation_filter=self._conversation_filter,
            category_filter=self._category_filter,
        )
        selected_question_ids = {question.question_id for question in questions}
        return conversations, questions, audit_errors & selected_question_ids


def _sha256_file(path: Path) -> str:
    if not path.exists():
        raise BenchmarkLoadError("LoCoMo dataset file is missing.", path=str(path))

    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkLoadError(
            "LoCoMo dataset file could not be read.",
            path=str(path),
            error=str(exc),
        ) from exc
    return digest.hexdigest()


def _load_json(path: Path, *, artifact_name: str) -> object:
    if not path.exists():
        raise BenchmarkLoadError(f"{artifact_name} file is missing.", path=str(path))

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkLoadError(
            f"{artifact_name} file could not be read.",
            path=str(path),
            error=str(exc),
        ) from exc

    try:
        loaded: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise BenchmarkLoadError(
            f"{artifact_name} contains malformed JSON.",
            path=str(path),
            error=str(exc),
        ) from exc
    return loaded


def _load_audit_errors(path: Path) -> set[str]:
    artifact = _require_object(_load_json(path, artifact_name="LoCoMo audit artifact"), "audit")
    schema_version = _require_int(artifact.get("schema_version"), "audit.schema_version")
    if schema_version != 1:
        raise BenchmarkLoadError(
            "LoCoMo audit artifact has an unsupported schema version.",
            path=str(path),
            schema_version=schema_version,
        )

    source = _require_object(artifact.get("source"), "audit.source")
    included_count = _require_int(source.get("included_count"), "audit.source.included_count")
    raw_errors = _require_list(artifact.get("errors"), "audit.errors")
    if included_count != len(raw_errors):
        raise BenchmarkLoadError(
            "LoCoMo audit artifact included_count does not match errors length.",
            path=str(path),
            included_count=included_count,
            errors_count=len(raw_errors),
        )

    question_ids: set[str] = set()
    for error_index, raw_error in enumerate(raw_errors):
        error_context = f"audit.errors[{error_index}]"
        error = _require_object(raw_error, error_context)
        question_id = _require_str(error.get("question_id"), f"{error_context}.question_id")
        error_type = _require_str(error.get("error_type"), f"{error_context}.error_type")
        if error_type == "WRONG_CITATION":
            raise BenchmarkLoadError(
                "LoCoMo audit artifact must exclude WRONG_CITATION entries.",
                path=str(path),
                question_id=question_id,
            )
        if question_id in question_ids:
            raise BenchmarkLoadError(
                "LoCoMo audit artifact contains a duplicate question ID.",
                path=str(path),
                question_id=question_id,
            )
        question_ids.add(question_id)
    return question_ids


def _parse_dataset(
    dataset: object,
    audit_errors: set[str],
    *,
    include_image_descriptions: bool = False,
) -> tuple[list[Conversation], list[Question]]:
    samples = _require_list(dataset, "dataset")
    conversations: list[Conversation] = []
    questions: list[Question] = []

    for conversation_index, raw_sample in enumerate(samples):
        sample = _require_object(raw_sample, f"dataset[{conversation_index}]")
        conversation_id = (
            _get_optional_str(sample, "sample_id") or f"locomo_conv{conversation_index}"
        )
        conversation_payload = _require_object(
            sample.get("conversation"),
            f"dataset[{conversation_index}].conversation",
        )
        conversations.append(
            _parse_conversation(
                conversation_payload,
                conversation_id=conversation_id,
                conversation_index=conversation_index,
                include_image_descriptions=include_image_descriptions,
            )
        )
        questions.extend(
            _parse_questions(
                sample,
                conversation_id=conversation_id,
                conversation_index=conversation_index,
                audit_errors=audit_errors,
            )
        )

    return conversations, questions


def _optional_str_filter(value: object, field: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise BenchmarkLoadError(
            "LoCoMo filter must be a list of strings.",
            field=field,
            observed_type=type(value).__name__,
        )

    raw_items = cast(list[object], value)
    items: list[str] = []
    for item_index, item in enumerate(raw_items):
        if not isinstance(item, str) or not item:
            raise BenchmarkLoadError(
                "LoCoMo filter values must be non-empty strings.",
                field=f"{field}[{item_index}]",
                observed_type=type(item).__name__,
            )
        items.append(item)
    return items


def _optional_category_filter(value: object) -> list[str] | None:
    categories = _optional_str_filter(value, "categories")
    if categories is None:
        return None

    allowed_categories = {category.value for category in QuestionCategory}
    unknown_categories = set(categories) - allowed_categories
    if unknown_categories:
        raise BenchmarkLoadError(
            "LoCoMo category filter contains unknown category values.",
            categories=sorted(unknown_categories),
            allowed_categories=sorted(allowed_categories),
        )
    return categories


def _apply_configured_filters(
    conversations: list[Conversation],
    questions: list[Question],
    *,
    conversation_filter: list[str] | None,
    category_filter: list[str] | None,
) -> tuple[list[Conversation], list[Question]]:
    selected_conversations = list(conversations)
    selected_questions = list(questions)

    if conversation_filter is not None:
        all_conversation_ids = {conversation.conversation_id for conversation in conversations}
        unknown_conversation_ids = set(conversation_filter) - all_conversation_ids
        if unknown_conversation_ids:
            raise BenchmarkLoadError(
                "LoCoMo conversation filter contains unknown conversation IDs.",
                conversation_ids=sorted(unknown_conversation_ids),
                allowed_conversation_ids=sorted(all_conversation_ids),
            )

        conversation_ids = set(conversation_filter)
        selected_conversations = [
            conversation
            for conversation in selected_conversations
            if conversation.conversation_id in conversation_ids
        ]
        if not selected_conversations:
            raise BenchmarkLoadError(
                "LoCoMo conversation filter selected no conversations.",
                conversations=conversation_filter,
            )
        selected_questions = [
            question
            for question in selected_questions
            if question.conversation_id in conversation_ids
        ]

    if category_filter is not None:
        category_values = set(category_filter)
        selected_questions = [
            question
            for question in selected_questions
            if question.category.value in category_values
        ]

    if not selected_questions:
        raise BenchmarkLoadError(
            "LoCoMo filters selected no questions.",
            conversations=conversation_filter,
            categories=category_filter,
        )

    conversation_ids_with_questions = {question.conversation_id for question in selected_questions}
    selected_conversations = [
        conversation
        for conversation in selected_conversations
        if conversation.conversation_id in conversation_ids_with_questions
    ]
    if not selected_conversations:
        raise BenchmarkLoadError(
            "LoCoMo filters selected no conversations.",
            conversations=conversation_filter,
            categories=category_filter,
        )

    return selected_conversations, selected_questions


def _parse_conversation(
    conversation_payload: dict[str, object],
    *,
    conversation_id: str,
    conversation_index: int,
    include_image_descriptions: bool = False,
) -> Conversation:
    speaker_a = _require_str(
        conversation_payload.get("speaker_a"),
        f"dataset[{conversation_index}].conversation.speaker_a",
    )
    speaker_b = _require_str(
        conversation_payload.get("speaker_b"),
        f"dataset[{conversation_index}].conversation.speaker_b",
    )
    sessions = [
        _parse_session(
            conversation_payload,
            conversation_id=conversation_id,
            conversation_index=conversation_index,
            session_number=session_number,
            include_image_descriptions=include_image_descriptions,
        )
        for session_number in _session_numbers(conversation_payload)
    ]
    return Conversation(
        conversation_id=conversation_id,
        speakers=[speaker_a, speaker_b],
        sessions=sessions,
    )


def _session_numbers(conversation_payload: dict[str, object]) -> list[int]:
    session_numbers: list[int] = []
    for key, value in conversation_payload.items():
        match = _SESSION_KEY_RE.fullmatch(key)
        if match is not None and value is not None:
            session_numbers.append(int(match.group(1)))
    return sorted(session_numbers)


def _parse_session(
    conversation_payload: dict[str, object],
    *,
    conversation_id: str,
    conversation_index: int,
    include_image_descriptions: bool = False,
    session_number: int,
) -> Session:
    session_key = f"session_{session_number}"
    timestamp_key = f"{session_key}_date_time"
    timestamp_text = _require_str(
        conversation_payload.get(timestamp_key),
        f"dataset[{conversation_index}].conversation.{timestamp_key}",
    )
    raw_turns = _require_list(
        conversation_payload.get(session_key),
        f"dataset[{conversation_index}].conversation.{session_key}",
    )
    turns = [
        _parse_turn(
            raw_turn,
            turn_context=(
                f"dataset[{conversation_index}].conversation.{session_key}[{turn_index}]"
            ),
            include_image_descriptions=include_image_descriptions,
        )
        for turn_index, raw_turn in enumerate(raw_turns)
    ]
    return Session(
        session_id=f"{conversation_id}_{session_key}",
        session_number=session_number,
        timestamp=_parse_timestamp(timestamp_text),
        turns=turns,
    )


def _parse_timestamp(timestamp_text: str) -> datetime:
    normalized = re.sub(r"\b(am|pm)\b", lambda match: match.group(1).upper(), timestamp_text)
    try:
        parsed = datetime.strptime(normalized, "%I:%M %p on %d %B, %Y")
    except ValueError as exc:
        raise BenchmarkLoadError(
            "LoCoMo session timestamp could not be parsed.",
            timestamp=timestamp_text,
        ) from exc
    return parsed.replace(tzinfo=UTC)


def _parse_turn(
    raw_turn: object,
    *,
    turn_context: str,
    include_image_descriptions: bool = False,
) -> Turn:
    turn = _require_object(raw_turn, turn_context)
    metadata = {field: turn[field] for field in _OPTIONAL_TURN_METADATA_FIELDS if field in turn}
    content = _require_str(turn.get("text"), f"{turn_context}.text")
    if include_image_descriptions:
        content = _with_image_description(content, metadata)
    return Turn(
        turn_id=_require_str(turn.get("dia_id"), f"{turn_context}.dia_id"),
        speaker=_require_str(turn.get("speaker"), f"{turn_context}.speaker"),
        content=content,
        metadata=metadata,
    )


def _with_image_description(content: str, metadata: dict[str, object]) -> str:
    """Append what an image-sharing turn actually showed, in the reference's own format.

    1,226 of LoCoMo's 5,882 turns -- 20.8% -- carry a `blip_caption` describing an image the
    speaker shared. Khedron parsed it into metadata and ingested only the spoken text, so the memory
    store never held a fifth of the corpus and questions about what a photo showed were unanswerable
    by construction.

    Khedron's own rendering, and only the caption. The dataset also carries `query` -- what
    its authors searched to find the image -- which is construction scaffolding rather than anything
    a speaker said, and it contradicts the caption systematically: "painting sunrise" against *a
    painting of a sunset*, "transgender pride flag mural" against *a dog walking past a wall*.
    Ingesting both would put two incompatible descriptions of one image into memory. It stays in
    `Turn.metadata` for audit and never reaches the store.

    Off by default. Turning it on changes what the memory store contains, so every score measured
    without it belongs to a different corpus -- which is why the profile pins the policy by name and
    it enters the fingerprint rather than being a quiet improvement.
    """
    caption = metadata.get("blip_caption")
    if not isinstance(caption, str) or not caption.strip():
        return content
    return f"{content} [shared an image depicting: {caption.strip()}]"


def _parse_questions(
    sample: dict[str, object],
    *,
    conversation_id: str,
    conversation_index: int,
    audit_errors: set[str],
) -> list[Question]:
    raw_questions = _require_list(sample.get("qa"), f"dataset[{conversation_index}].qa")
    sample_id = _get_optional_str(sample, "sample_id")
    questions: list[Question] = []
    for qa_index, raw_question in enumerate(raw_questions):
        question_context = f"dataset[{conversation_index}].qa[{qa_index}]"
        question_payload = _require_object(raw_question, question_context)
        raw_category = _require_int(
            question_payload.get("category"), f"{question_context}.category"
        )
        category = _category_from_raw(raw_category, question_context)
        question_id = f"locomo_{conversation_index}_qa{qa_index}"
        expected_answer, answer_metadata = _answer_metadata(
            question_payload,
            question_context=question_context,
            category=category,
        )
        metadata: dict[str, Any] = {
            "sample_id": sample_id,
            "conversation_index": conversation_index,
            "qa_index": qa_index,
            "raw_category": raw_category,
            **answer_metadata,
        }
        if "adversarial_answer" in question_payload:
            metadata["adversarial_answer"] = question_payload["adversarial_answer"]

        questions.append(
            Question(
                question_id=question_id,
                conversation_id=conversation_id,
                category=category,
                question_text=_require_str(
                    question_payload.get("question"),
                    f"{question_context}.question",
                ),
                expected_answer=expected_answer,
                evidence_dialog_ids=_require_str_list(
                    question_payload.get("evidence"),
                    f"{question_context}.evidence",
                ),
                is_audited_error=question_id in audit_errors,
                metadata=metadata,
            )
        )
    return questions


def _answer_metadata(
    question_payload: dict[str, object],
    *,
    question_context: str,
    category: QuestionCategory,
) -> tuple[str, dict[str, Any]]:
    raw_answer = question_payload.get("answer")
    if raw_answer is None:
        if category is not QuestionCategory.ADVERSARIAL:
            raise BenchmarkLoadError(
                "LoCoMo non-adversarial question is missing a ground-truth answer.",
                question=question_context,
                category=category.value,
            )
        return "", {"answerable": False, "raw_answer": None, "raw_answer_is_null": True}

    return str(raw_answer), {
        "answerable": True,
        "raw_answer": raw_answer,
        "raw_answer_is_null": False,
    }


def _category_from_raw(raw_category: int, question_context: str) -> QuestionCategory:
    try:
        return _CATEGORY_MAP[raw_category]
    except KeyError:
        raise BenchmarkLoadError(
            "LoCoMo question has an unknown category.",
            question=question_context,
            raw_category=raw_category,
        ) from None


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BenchmarkLoadError(
            "Expected JSON object.", field=context, observed_type=type(value).__name__
        )
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise BenchmarkLoadError(
            "Expected JSON list.", field=context, observed_type=type(value).__name__
        )
    return cast(list[object], value)


def _require_str(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkLoadError("Expected non-empty string.", field=context)
    return value


def _get_optional_str(mapping: dict[str, object], field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkLoadError(
            "Expected string.", field=field, observed_type=type(value).__name__
        )
    return value


def _require_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkLoadError(
            "Expected integer.", field=context, observed_type=type(value).__name__
        )
    return value


def _require_str_list(value: object, context: str) -> list[str]:
    raw_items = _require_list(value, context)
    items: list[str] = []
    for item_index, item in enumerate(raw_items):
        items.append(_require_str(item, f"{context}[{item_index}]"))
    return items
