from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

__all__ = ["ArchivedProfile", "archived_profile", "archived_profile_names"]

_ARCHIVE_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "methodology" / "archived_profiles.json"
)


@dataclass(frozen=True)
class ArchivedProfile:
    """A profile that ran, was removed, and must stay readable without being runnable.

    A tombstone, deliberately not a `MethodologyRuntimeProfile`. Reusing that type would mean
    carrying prompt paths for files that no longer exist -- validation checks they do -- and would
    let the runner resolve a name it must refuse. The two types differ because the two things
    differ: one is a contract a run can be executed under, this is a record that one was.

    It carries no prompt paths and recomputes no fingerprint. The fingerprint a run recorded is
    history and stands on its own; recomputing it from a definition that no longer exists would be
    inventing a value, and comparing against it would be comparing against an invention.
    """

    name: str
    archived_on: str
    reason: str
    # What the profile pinned, kept only so a report can still say what a run measured. Strings,
    # not live objects: nothing here is used to execute anything.
    answer_model_id: str | None
    judge_model_id: str | None
    evaluation_categories: tuple[str, ...] | None
    scored_categories: tuple[str, ...] | None

    @property
    def runnable(self) -> bool:
        """Always false. Present so the answer is stated rather than inferred from the type."""
        return False


@cache
def _archive() -> dict[str, ArchivedProfile]:
    if not _ARCHIVE_PATH.exists():
        return {}
    payload = json.loads(_ARCHIVE_PATH.read_text(encoding="utf-8"))
    return {
        entry["name"]: ArchivedProfile(
            name=entry["name"],
            archived_on=entry["archived_on"],
            reason=entry["reason"],
            answer_model_id=entry.get("answer_model_id"),
            judge_model_id=entry.get("judge_model_id"),
            evaluation_categories=_optional_tuple(entry.get("evaluation_categories")),
            scored_categories=_optional_tuple(entry.get("scored_categories")),
        )
        for entry in payload["profiles"]
    }


def _optional_tuple(value: list[str] | None) -> tuple[str, ...] | None:
    return None if value is None else tuple(value)


def archived_profile(name: str) -> ArchivedProfile | None:
    """The tombstone for a removed profile, or None when the name was never archived."""
    return _archive().get(name)


def archived_profile_names() -> frozenset[str]:
    return frozenset(_archive())
