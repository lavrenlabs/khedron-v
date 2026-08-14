from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from khedron import build
from khedron.build import UNIDENTIFIED_BUILD_SUFFIX, resolve_framework_version


def test_version_identifies_the_commit_and_flags_a_dirty_tree(monkeypatch: Any) -> None:
    # The point of the change: an artifact must name the build that produced it. A dirty tree is
    # recorded because a result produced from uncommitted code cannot be reproduced from any
    # commit -- that is a fact about the measurement, not a lint warning.
    calls: list[tuple[str, ...]] = []

    def fake(*args: str) -> str | None:
        calls.append(args)
        return "abc1234" if args[0] == "rev-parse" else " M src/khedron/runner.py"

    monkeypatch.setattr(build, "_git_output", fake)
    monkeypatch.setattr(build, "_declared_version", lambda: "1.2.3")

    if resolve_framework_version() != "1.2.3+abc1234.dirty":
        raise AssertionError(resolve_framework_version())
    if calls[0][0] != "rev-parse":
        raise AssertionError(calls)


def test_a_clean_tree_records_the_commit_without_a_dirty_marker(monkeypatch: Any) -> None:
    monkeypatch.setattr(build, "_git_output", lambda *a: "abc1234" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(build, "_declared_version", lambda: "1.2.3")

    if resolve_framework_version() != "1.2.3+abc1234":
        raise AssertionError(resolve_framework_version())


def test_unknown_cleanliness_is_recorded_as_dirty(monkeypatch: Any) -> None:
    # The status call failing where rev-parse succeeded leaves cleanliness unverified. Claiming
    # a clean tree we did not confirm would overstate the provenance, which is the exact class
    # of error this change exists to remove.
    monkeypatch.setattr(build, "_git_output", lambda *a: "abc1234" if a[0] == "rev-parse" else None)
    monkeypatch.setattr(build, "_declared_version", lambda: "1.2.3")

    if resolve_framework_version() != "1.2.3+abc1234.dirty":
        raise AssertionError(resolve_framework_version())


def test_absent_git_says_so_instead_of_returning_a_bare_version(monkeypatch: Any) -> None:
    # "0.0.0" reads as a release. The whole defect was a version that identified no build while
    # looking like one, so the failure path has to be legible in the artifact.
    monkeypatch.setattr(build, "_git_output", lambda *a: None)
    monkeypatch.setattr(build, "_declared_version", lambda: "0.0.0")

    resolved = resolve_framework_version()
    if resolved != f"0.0.0+{UNIDENTIFIED_BUILD_SUFFIX}":
        raise AssertionError(resolved)


def test_resolution_never_raises_even_if_git_explodes(monkeypatch: Any) -> None:
    # Provenance resolution runs at the start of every run. It must not be able to fail one.
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("git is not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(build, "_declared_version", lambda: "0.0.0")

    if not resolve_framework_version().endswith(UNIDENTIFIED_BUILD_SUFFIX):
        raise AssertionError(resolve_framework_version())


def test_a_timeout_is_treated_as_unidentified(monkeypatch: Any) -> None:
    def slow(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="git", timeout=5.0)

    monkeypatch.setattr(subprocess, "run", slow)
    monkeypatch.setattr(build, "_declared_version", lambda: "0.0.0")

    if not resolve_framework_version().endswith(UNIDENTIFIED_BUILD_SUFFIX):
        raise AssertionError(resolve_framework_version())


def test_the_real_repository_resolves_to_a_commit() -> None:
    # Guards the wiring, not the logic: the tests above stub git out, so without this one the
    # suite would pass even if the subprocess call never worked here.
    resolved = resolve_framework_version()
    if UNIDENTIFIED_BUILD_SUFFIX in resolved:
        raise AssertionError(f"git resolution failed in the repository itself: {resolved}")
    if "+" not in resolved:
        raise AssertionError(resolved)


def test_resolution_describes_the_code_not_the_launch_directory(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The property the previous test does NOT prove. It runs from the repository root, so it
    # would pass even if `_git_output` omitted `cwd` entirely -- and then a run launched from
    # anywhere else would silently record `+unidentified`, or worse, another repository's commit.
    # Reviewed and found exactly that gap, which is why this test exists.
    monkeypatch.chdir(tmp_path)

    resolved = resolve_framework_version()

    if UNIDENTIFIED_BUILD_SUFFIX in resolved:
        raise AssertionError(
            f"resolution followed the launch directory instead of the source: {resolved}"
        )
    if "+" not in resolved:
        raise AssertionError(resolved)


def test_the_git_invocation_stays_argv_only_with_a_timeout(monkeypatch: Any) -> None:
    # Locks the shape of the call rather than only its result: no shell, a fixed argv, a bounded
    # timeout, and a cwd inside the package. A regression here is a security or hang risk that
    # every other test in this file would pass straight over.
    seen: dict[str, Any] = {}

    def capture(args: Any, **kwargs: Any) -> Any:
        seen["args"] = args
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="abc1234", stderr="")

    monkeypatch.setattr(subprocess, "run", capture)
    resolve_framework_version()

    if seen["args"][0] != "git" or not isinstance(seen["args"], list):
        raise AssertionError(seen["args"])
    if seen.get("shell"):
        raise AssertionError("git must never be invoked through a shell")
    if not seen.get("timeout"):
        raise AssertionError("the call must be bounded by a timeout")
    if seen.get("check") is not False:
        raise AssertionError("a non-zero exit is expected and handled, not raised")
    if Path(build.__file__).resolve().parent != Path(seen["cwd"]).resolve():
        raise AssertionError(f"cwd must be the package directory, got {seen['cwd']}")
