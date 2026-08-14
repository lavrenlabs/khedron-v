from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

CANONICAL_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
EXPECTED_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "data" / "locomo" / "locomo10.json"
CHUNK_SIZE_BYTES = 1024 * 1024


class DownloadLocomoError(RuntimeError):
    """Raised when the LoCoMo dataset cannot be downloaded or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, destination: Path) -> None:
    urllib.request.urlretrieve(url, str(destination))  # noqa: S310


def _temporary_destination(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _verify_expected_checksum(path: Path) -> str:
    observed = sha256_file(path)
    if observed != EXPECTED_SHA256:
        raise DownloadLocomoError(
            f"LoCoMo dataset checksum mismatch: expected {EXPECTED_SHA256}, observed {observed}."
        )
    return observed


def download_locomo(destination: Path, force: bool = False) -> Path:
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        observed = sha256_file(destination)
        if observed == EXPECTED_SHA256 and not force:
            print(f"LoCoMo dataset already present: {destination}")
            return destination
        if observed != EXPECTED_SHA256 and not force:
            raise DownloadLocomoError(
                "Existing LoCoMo dataset checksum mismatch: "
                f"expected {EXPECTED_SHA256}, observed {observed}. "
                "Use --force to replace it after verification."
            )

    temporary_path = _temporary_destination(destination)
    try:
        print(f"Downloading LoCoMo dataset from {CANONICAL_URL}")
        _download_url(CANONICAL_URL, temporary_path)
        observed = _verify_expected_checksum(temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"LoCoMo dataset saved to {destination} (sha256: {observed})")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the canonical LoCoMo dataset."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Destination path. Defaults to {DEFAULT_DESTINATION}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and replace the destination only after checksum verification.",
    )
    args = parser.parse_args(argv)

    try:
        download_locomo(args.destination, force=args.force)
    except (DownloadLocomoError, OSError, urllib.error.URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
