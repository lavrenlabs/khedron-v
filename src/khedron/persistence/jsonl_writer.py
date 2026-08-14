from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from khedron.utils.redaction import scrub_credentials


class JsonlWriter:
    """Append-only JSONL writer with buffering and async-safe flushing."""

    def __init__(self, path: Path, buffer_size: int = 100) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than or equal to 1")

        self._path = path
        self._buffer: list[str] = []
        self._buffer_size = buffer_size
        self._lock = asyncio.Lock()

    async def write(self, record: BaseModel) -> None:
        # Every persisted record passes through here, which is why the credential backstop
        # lives here rather than at each of the call sites that build a record. Six review
        # rounds each found another field or record that reached disk without one; per-site
        # policies remain the primary defence, and this catches what they have not been
        # taught about yet. It only removes what is unmistakably a credential -- these
        # records also carry the measurement.
        # ensure_ascii=False with compact separators reproduces `model_dump_json` byte for
        # byte. Plain `json.dumps` defaults inflated non-ASCII conversation text by about a
        # third and added spaces, changing the artifact format as a side effect of adding
        # redaction -- a gratuitous change to files that already exist on disk.
        line = (
            json.dumps(
                scrub_credentials(record.model_dump(mode="json")),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        async with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._buffer_size:
                await self._flush_locked()

    async def close(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return

        lines = list(self._buffer)
        await asyncio.to_thread(self._append_lines, self._path, lines)
        del self._buffer[: len(lines)]

    @staticmethod
    def _append_lines(path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as file:
            file.write("".join(lines).encode("utf-8"))
