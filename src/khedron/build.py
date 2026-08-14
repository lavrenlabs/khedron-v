from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

__all__ = ["UNIDENTIFIED_BUILD_SUFFIX", "is_unidentifiable_build", "resolve_framework_version"]

_PACKAGE: Final = "khedron"
_PLACEHOLDER_VERSION: Final = "0.0.0"
_GIT_TIMEOUT_SECONDS: Final = 5.0

# Appended when the build cannot be identified, instead of returning a bare version that looks
# like an answer. An artifact that says "0.0.0" reads as a release; one that says
# "0.0.0+unidentified" says out loud that its provenance is unknown.
UNIDENTIFIED_BUILD_SUFFIX: Final = "unidentified"


def is_unidentifiable_build(version: str) -> bool:
    """True when a version string names no build a commit can reproduce.

    The single predicate behind all three build-identity guards -- preflight's advisory blocker
    (``preflight._check_build_is_identifiable``), the Runner execution-boundary refusal, and
    resume's unconditional refusal (``resume._require_resumable``) -- so they test build identity
    one way and cannot drift apart. A version ends ``.dirty`` (produced from an uncommitted tree)
    or ``+unidentified`` (the source was not a git checkout); either way no commit reproduces it,
    so no result built on it can state what measured it.
    """
    return version.endswith(".dirty") or version.endswith(f"+{UNIDENTIFIED_BUILD_SUFFIX}")


def resolve_framework_version() -> str:
    """Identify the build that produced a run, not merely its declared version.

    Why this exists: `pyproject.toml` carries the placeholder ``0.0.0`` and nothing else records
    a commit, so every artifact named a version that identifies no build. The methodology
    promises a third party can reproduce a result from the recorded provenance; a placeholder
    makes that promise unkeepable.

    Why git rather than a build-time stamp: Khedron runs from a checkout and is not distributed
    as an artifact, so the commit is available at run time exactly when a run happens. A
    packaging-time stamp would solve a distribution problem this project does not have, and
    would leave the common case -- running from the working tree -- still unidentified.

    The dirty marker is not decoration. A benchmark result produced from an uncommitted tree
    cannot be reproduced from any commit, and that is a fact about the measurement rather than
    a lint warning, so it belongs in the recorded version string.

    Never raises: provenance resolution must not be able to fail a run that is already spending
    money. When git is absent or the source is not in a repository, the returned string says so.
    """
    base = _declared_version()
    commit = _git_output("rev-parse", "--short", "HEAD")
    if commit is None:
        return f"{base}+{UNIDENTIFIED_BUILD_SUFFIX}"
    status = _git_output("status", "--porcelain")
    # `status` of None means the second call failed where the first succeeded; treat unknown
    # cleanliness as dirty rather than claim a clean tree we did not verify.
    if status is None or status:
        return f"{base}+{commit}.dirty"
    return f"{base}+{commit}"


def _declared_version() -> str:
    try:
        return version(_PACKAGE)
    except PackageNotFoundError:
        return _PLACEHOLDER_VERSION


def _git_output(*args: str) -> str | None:
    """Return trimmed git output, or None if git cannot answer.

    Runs in the directory of this source file, so the answer describes the code that is
    executing rather than whatever directory the operator happened to launch from.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["git", *args],  # noqa: S607 - resolved from PATH deliberately; no vendored git
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # git missing, not executable, or timed out.
        return None
    if completed.returncode != 0:
        # Most often: the source is not inside a git repository.
        return None
    return completed.stdout.strip()
