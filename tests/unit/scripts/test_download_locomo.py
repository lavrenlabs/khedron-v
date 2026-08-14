from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_downloader_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "download_locomo.py"
    spec = importlib.util.spec_from_file_location("download_locomo_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


downloader = load_downloader_module()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_sha256_file_returns_expected_digest_for_known_bytes(tmp_path: Path) -> None:
    payload = b"khedron"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    observed = downloader.sha256_file(path)

    if observed != digest(payload):
        raise AssertionError(observed)


def test_existing_valid_destination_skips_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"canonical locomo"
    destination = tmp_path / "locomo10.json"
    destination.write_bytes(payload)
    monkeypatch.setattr(downloader, "EXPECTED_SHA256", digest(payload))

    def fail_download(_url: str, _destination: Path) -> None:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(downloader, "_download_url", fail_download)

    result = downloader.download_locomo(destination)

    if result != destination:
        raise AssertionError(result)
    if destination.read_bytes() != payload:
        raise AssertionError(destination.read_bytes())


def test_existing_invalid_destination_fails_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "locomo10.json"
    destination.write_bytes(b"stale")
    monkeypatch.setattr(downloader, "EXPECTED_SHA256", digest(b"canonical locomo"))

    with pytest.raises(downloader.DownloadLocomoError, match="Use --force"):
        downloader.download_locomo(destination)


def test_force_replaces_invalid_destination_after_mocked_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"canonical locomo"
    destination = tmp_path / "locomo10.json"
    destination.write_bytes(b"stale")
    monkeypatch.setattr(downloader, "EXPECTED_SHA256", digest(payload))

    def write_payload(_url: str, download_destination: Path) -> None:
        download_destination.write_bytes(payload)

    monkeypatch.setattr(downloader, "_download_url", write_payload)

    result = downloader.download_locomo(destination, force=True)

    if result != destination:
        raise AssertionError(result)
    if destination.read_bytes() != payload:
        raise AssertionError(destination.read_bytes())


def test_downloaded_checksum_mismatch_does_not_replace_existing_valid_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_payload = b"canonical locomo"
    destination = tmp_path / "locomo10.json"
    destination.write_bytes(existing_payload)
    monkeypatch.setattr(downloader, "EXPECTED_SHA256", digest(existing_payload))

    def write_invalid_payload(_url: str, download_destination: Path) -> None:
        download_destination.write_bytes(b"not the expected dataset")

    monkeypatch.setattr(downloader, "_download_url", write_invalid_payload)

    with pytest.raises(downloader.DownloadLocomoError, match="checksum mismatch"):
        downloader.download_locomo(destination, force=True)

    if destination.read_bytes() != existing_payload:
        raise AssertionError(destination.read_bytes())


def test_main_returns_zero_for_already_present_valid_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"canonical locomo"
    destination = tmp_path / "locomo10.json"
    destination.write_bytes(payload)
    monkeypatch.setattr(downloader, "EXPECTED_SHA256", digest(payload))

    def fail_download(_url: str, _destination: Path) -> None:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(downloader, "_download_url", fail_download)

    exit_code = downloader.main(["--destination", str(destination)])

    if exit_code != 0:
        raise AssertionError(exit_code)
