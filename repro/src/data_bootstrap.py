"""Download and verify the exact public datasets used by the baseline.

The experiment runner starts in a fresh clone, so ignored raw data cannot be
assumed to exist. Every downloaded byte is checked against the committed
manifest before it becomes visible at its target path.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "repro" / "audit" / "public_data_manifest.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_valid(path: Path, expected: str) -> bool:
    return path.is_file() and _sha256_bytes(path.read_bytes()) == expected


def _download(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _write_verified(target: Path, data: bytes, expected: str) -> None:
    actual = _sha256_bytes(data)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {target}: expected {expected}, observed {actual}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(target)


def ensure_public_data(root: Path = ROOT) -> None:
    manifest = json.loads(MANIFEST.read_text())
    user_agent = manifest["user_agent"]
    downloaded = 0

    for entry in manifest["files"]:
        target = root / entry["target"]
        if _is_valid(target, entry["sha256"]):
            continue
        _write_verified(
            target,
            _download(entry["url"], user_agent),
            entry["sha256"],
        )
        downloaded += 1

    for archive_entry in manifest["archives"]:
        needed = [
            member
            for member in archive_entry["members"]
            if not _is_valid(root / member["target"], member["sha256"])
        ]
        if not needed:
            continue
        archive_bytes = _download(archive_entry["url"], user_agent)
        actual = _sha256_bytes(archive_bytes)
        if actual != archive_entry["sha256"]:
            raise ValueError(
                "Archive SHA-256 mismatch: "
                f"expected {archive_entry['sha256']}, observed {actual}"
            )
        with tempfile.NamedTemporaryFile(suffix=".zip") as handle:
            handle.write(archive_bytes)
            handle.flush()
            with zipfile.ZipFile(handle.name) as archive:
                available = set(archive.namelist())
                for member in needed:
                    if member["member"] not in available:
                        raise ValueError(f"Missing archive member: {member['member']}")
                    _write_verified(
                        root / member["target"],
                        archive.read(member["member"]),
                        member["sha256"],
                    )
                    downloaded += 1

    verified = len(manifest["files"]) + sum(
        len(entry["members"]) for entry in manifest["archives"]
    )
    print(f"DATA_BOOTSTRAP verified={verified} downloaded={downloaded}")


if __name__ == "__main__":
    ensure_public_data()
