from __future__ import annotations

import ssl
from pathlib import Path

import pytest

import khedron.tls as tls_module
from khedron.tls import configure_os_trust_store


def test_configure_os_trust_store_routes_ssl_through_os_store_idempotently() -> None:
    import truststore

    tls_module._configured = False
    try:
        if configure_os_trust_store() is not True:
            raise AssertionError("expected the OS trust store to be enabled")
        if ssl.SSLContext is not truststore.SSLContext:
            raise AssertionError("ssl.SSLContext was not routed through truststore")
        # A second call is a no-op that still reports success.
        if configure_os_trust_store() is not True:
            raise AssertionError("second call did not report success")
    finally:
        truststore.extract_from_ssl()
        tls_module._configured = False


def test_live_api_entry_points_configure_the_trust_store() -> None:
    # Entry points that build SDK clients outside the Typer CLI must enable
    # OS-trust-store verification themselves, or they fail under HTTPS inspection even
    # though the CLI path works. Guards the CLI callback and the real-API smoke script.
    cli_source = (Path(__file__).resolve().parents[2] / "src" / "khedron" / "cli.py").read_text(
        encoding="utf-8"
    )
    if "configure_os_trust_store()" not in cli_source:
        raise AssertionError("khedron CLI does not configure the OS trust store")

    smoke_source = (
        Path(__file__).resolve().parents[2] / "scripts" / "smoke_phase4_apis.py"
    ).read_text(encoding="utf-8")
    if "configure_os_trust_store()" not in smoke_source:
        raise AssertionError("real-API smoke script does not configure the OS trust store")


def test_configure_os_trust_store_is_best_effort_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truststore

    def _raise() -> None:
        raise RuntimeError("simulated injection failure")

    tls_module._configured = False
    monkeypatch.setattr(truststore, "inject_into_ssl", _raise)
    try:
        # A failure must be swallowed (certifi stays in place) and reported, never raised.
        if configure_os_trust_store() is not False:
            raise AssertionError("expected False when injection fails")
    finally:
        tls_module._configured = False


def test_trust_store_fallback_notice_never_touches_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The CLI runs this before every command and documented automation parses stdout
    # (e.g. `inspect suite --field total_cost_usd`), so the fallback notice must go to
    # stderr only; an extra stdout line would corrupt machine-readable output.
    import truststore

    def _raise() -> None:
        raise RuntimeError("simulated injection failure")

    tls_module._configured = False
    monkeypatch.setattr(truststore, "inject_into_ssl", _raise)
    try:
        configure_os_trust_store()
    finally:
        tls_module._configured = False

    captured = capsys.readouterr()
    if captured.out != "":
        raise AssertionError(f"stdout must stay clean, got: {captured.out!r}")
    if "trust store" not in captured.err.lower():
        raise AssertionError(f"expected the notice on stderr, got: {captured.err!r}")
