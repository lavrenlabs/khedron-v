from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from khedron.persistence.jsonl_writer import JsonlWriter


class SyntheticRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    sequence_number: int
    payload: str


def record(sequence_number: int, *, payload: str = "value") -> SyntheticRecord:
    return SyntheticRecord(
        record_id=f"record-{sequence_number}",
        sequence_number=sequence_number,
        payload=payload,
    )


async def read_bytes(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)


async def read_lines(path: Path) -> list[str]:
    content = (await read_bytes(path)).decode("utf-8")
    return content.splitlines()


async def exists(path: Path) -> bool:
    return await asyncio.to_thread(path.exists)


@pytest.mark.asyncio
async def test_write_creates_one_valid_json_line_per_record(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [record(0), record(1, payload="caf\u00e9"), record(2)]
    writer = JsonlWriter(path, buffer_size=10)

    for item in records:
        await writer.write(item)
    await writer.close()

    raw_content = await read_bytes(path)
    lines = raw_content.decode("utf-8").splitlines()

    if raw_content.count(b"\n") != len(records):
        raise AssertionError(raw_content)
    if b"\r\n" in raw_content:
        raise AssertionError(raw_content)
    if len(lines) != len(records):
        raise AssertionError(lines)

    parsed = [SyntheticRecord.model_validate_json(line) for line in lines]
    if parsed != records:
        raise AssertionError(parsed)
    for line in lines:
        if not isinstance(json.loads(line), dict):
            raise AssertionError(line)


@pytest.mark.asyncio
async def test_close_flushes_records_that_have_not_reached_buffer_size(tmp_path: Path) -> None:
    path = tmp_path / "buffered.jsonl"
    writer = JsonlWriter(path, buffer_size=10)

    await writer.write(record(0))
    await writer.write(record(1))

    if await exists(path):
        lines_before_close = await read_lines(path)
        if lines_before_close != []:
            raise AssertionError(lines_before_close)

    await writer.close()

    lines_after_close = await read_lines(path)
    if len(lines_after_close) != 2:
        raise AssertionError(lines_after_close)


@pytest.mark.asyncio
async def test_automatic_flush_happens_at_buffer_size(tmp_path: Path) -> None:
    path = tmp_path / "auto-flush.jsonl"
    writer = JsonlWriter(path, buffer_size=2)

    await writer.write(record(0))
    if await exists(path):
        lines_before_threshold = await read_lines(path)
        if lines_before_threshold != []:
            raise AssertionError(lines_before_threshold)

    await writer.write(record(1))

    lines_after_threshold = await read_lines(path)
    if len(lines_after_threshold) != 2:
        raise AssertionError(lines_after_threshold)

    await writer.close()
    lines_after_close = await read_lines(path)
    if lines_after_close != lines_after_threshold:
        raise AssertionError(lines_after_close)


@pytest.mark.asyncio
async def test_concurrent_writes_produce_complete_valid_lines(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    writer = JsonlWriter(path, buffer_size=7)
    expected_records = [record(sequence_number) for sequence_number in range(50)]

    await asyncio.gather(*(writer.write(item) for item in expected_records))
    await writer.close()

    lines = await read_lines(path)
    parsed = [SyntheticRecord.model_validate_json(line) for line in lines]

    if len(lines) != len(expected_records):
        raise AssertionError(lines)
    if {item.record_id for item in parsed} != {item.record_id for item in expected_records}:
        raise AssertionError(parsed)
    for line in lines:
        if not isinstance(json.loads(line), dict):
            raise AssertionError(line)


@pytest.mark.asyncio
async def test_parent_directories_are_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run" / "records.jsonl"
    writer = JsonlWriter(path, buffer_size=1)

    await writer.write(record(0))
    await writer.close()

    if not await exists(path):
        raise AssertionError(path)
    lines = await read_lines(path)
    if len(lines) != 1:
        raise AssertionError(lines)


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "idempotent.jsonl"
    writer = JsonlWriter(path, buffer_size=10)

    await writer.write(record(0))
    await writer.close()
    await writer.close()

    lines = await read_lines(path)
    if len(lines) != 1:
        raise AssertionError(lines)


@pytest.mark.parametrize("buffer_size", [0, -1])
def test_invalid_buffer_size_fails_fast(tmp_path: Path, buffer_size: int) -> None:
    with pytest.raises(ValueError, match="buffer_size"):
        JsonlWriter(tmp_path / "invalid.jsonl", buffer_size=buffer_size)


@pytest.mark.asyncio
async def test_writer_scrubs_unmistakable_credentials_from_any_record(tmp_path: Path) -> None:
    # The backstop at the persistence boundary. Six review rounds each found another field or
    # record that reached disk without a policy of its own; this catches a record type nobody
    # has taught a policy about yet -- SyntheticRecord has none.
    path = tmp_path / "records.jsonl"
    writer = JsonlWriter(path, buffer_size=1)

    await writer.write(record(1, payload="sk-proj-must-never-reach-disk"))
    await writer.close()

    written = path.read_text(encoding="utf-8")
    if "sk-proj-must-never-reach-disk" in written:
        raise AssertionError(written)
    if "[REDACTED]" not in written:
        raise AssertionError(written)


@pytest.mark.asyncio
async def test_writer_keeps_measurement_text_that_resembles_an_auth_scheme(
    tmp_path: Path,
) -> None:
    # The bound on that backstop, chosen deliberately: these records also carry the corpus, the
    # generated answers and the judge rationales. "Basic ..." and "Bearer ..." begin English
    # sentences far more often than auth headers, so they are excluded from the boundary scan
    # and left to the field-level policies. Redacting a measurement to protect it would defeat
    # the purpose of recording it.
    path = tmp_path / "records.jsonl"
    writer = JsonlWriter(path, buffer_size=1)

    await writer.write(record(1, payload="Basic recall of the first session"))
    await writer.write(record(2, payload="Bearer in mind the June conversation"))
    await writer.close()

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    payloads = [line["payload"] for line in lines]
    if payloads != ["Basic recall of the first session", "Bearer in mind the June conversation"]:
        raise AssertionError(payloads)


@pytest.mark.asyncio
async def test_writer_renders_non_finite_floats_as_null(tmp_path: Path) -> None:
    # Regression guard for the scrub itself. `model_dump_json` renders a non-finite float as
    # null; `json.dumps(model_dump(mode="json"))` emits bare `Infinity`, which strict JSON
    # parsers reject -- so adding the credential scrub silently degraded the artifact format
    # until this was pinned. Redaction must not change what the records mean.
    class FloatRecord(BaseModel):
        model_config = ConfigDict(frozen=True)

        ratio: float

    path = tmp_path / "floats.jsonl"
    writer = JsonlWriter(path, buffer_size=1)

    await writer.write(FloatRecord(ratio=float("inf")))
    await writer.close()

    written = path.read_text(encoding="utf-8").strip()
    if "Infinity" in written:
        raise AssertionError(written)
    if json.loads(written)["ratio"] is not None:
        raise AssertionError(written)


@pytest.mark.asyncio
async def test_writer_output_is_byte_identical_to_pydantic_serialization(tmp_path: Path) -> None:
    # The scrub replaced model_dump_json with json.dumps, whose defaults escape non-ASCII and
    # add spaces -- inflating conversation text by roughly a third and changing the format of
    # files that already exist on disk. Adding redaction must not restyle the artifacts.
    path = tmp_path / "unicode.jsonl"
    writer = JsonlWriter(path, buffer_size=1)
    item = record(1, payload="café — sesión de junio")

    await writer.write(item)
    await writer.close()

    written = path.read_text(encoding="utf-8").rstrip("\n")
    if written != item.model_dump_json():
        raise AssertionError(f"{written!r} != {item.model_dump_json()!r}")
