#!/usr/bin/env python3
"""Validate the additive, text-only Hugging Face Space release candidate."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / ".trackio" / "logbook"
PROTECTED_MANIFEST = ROOT / "repro" / "audit" / "judged_space_manifest.sha256"
ALLOWLIST = ROOT / "release" / "hf-space-upload-allowlist.txt"
CANDIDATE_MANIFEST = ROOT / "release" / "hf-space-candidate.sha256"
SUBSET_REPORT = ROOT / "release" / "hf-space-subset-check.json"
MUTABLE_NAVIGATION = {"logbook.json", "pages/index.md"}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "Bearer credential": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protected_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in PROTECTED_MANIFEST.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        digest, path = line.split("  ", 1)
        entries[path] = digest
    return entries


def candidate_files() -> dict[str, Path]:
    return {
        str(path.relative_to(CANDIDATE)): path
        for path in sorted(CANDIDATE.rglob("*"))
        if path.is_file()
    }


def validate_navigation(files: dict[str, Path]) -> tuple[list[str], list[str]]:
    payload = json.loads((CANDIDATE / "logbook.json").read_text())
    if payload["space_id"] != "DineshAI/KJq0iScNM6":
        raise ValueError("candidate targets the wrong Space")
    children = payload["root"]["children"]
    slugs = [child["slug"] for child in children]
    if len(slugs) != len(set(slugs)):
        raise ValueError("duplicate sidebar slug")
    referenced = [payload["root"]["file"]]
    referenced.extend(child["file"] for child in children)
    missing = [path for path in referenced if path not in files]
    if missing:
        raise ValueError(f"sidebar references missing files: {missing}")
    index = (CANDIDATE / "pages" / "index.md").read_text()
    index_slugs = re.findall(r"\(#/([A-Za-z0-9._-]+)\)", index)
    if index_slugs != slugs:
        raise ValueError("index page order differs from logbook.json")
    return slugs, referenced


def validate_allowlist(files: dict[str, Path]) -> list[str]:
    paths = [
        line.strip()
        for line in ALLOWLIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("upload allowlist contains duplicates")
    missing = [path for path in paths if path not in files]
    if missing:
        raise ValueError(f"upload allowlist contains missing files: {missing}")
    for relative in paths:
        path = files[relative]
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"allowlisted path is not UTF-8 text: {relative}") from error
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"{label} pattern found in {relative}")
    return paths


def main() -> None:
    protected = read_protected_manifest()
    files = candidate_files()
    missing_old_paths = sorted(set(protected) - set(files))
    if missing_old_paths:
        raise ValueError(f"candidate removed judged paths: {missing_old_paths}")
    changed_old_paths = sorted(
        path for path, digest in protected.items() if sha256(files[path]) != digest
    )
    unexpected_changes = sorted(set(changed_old_paths) - MUTABLE_NAVIGATION)
    if unexpected_changes:
        raise ValueError(
            f"judged evidence changed outside navigation files: {unexpected_changes}"
        )
    slugs, referenced = validate_navigation(files)
    allowlist = validate_allowlist(files)
    manifest_lines = [
        f"{sha256(path)}  {relative}" for relative, path in sorted(files.items())
    ]
    CANDIDATE_MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    report = {
        "judged_revision": "e57f7e6e348fea6c5a0467ca33f94375b5bf2623",
        "space_id": "DineshAI/KJq0iScNM6",
        "old_path_count": len(protected),
        "candidate_path_count": len(files),
        "old_paths_subset_of_candidate": not missing_old_paths,
        "byte_identical_old_path_count": len(protected) - len(changed_old_paths),
        "changed_navigation_paths": changed_old_paths,
        "changed_old_evidence_paths": unexpected_changes,
        "sidebar_slugs": slugs,
        "referenced_page_count": len(referenced),
        "text_only_upload_allowlist": allowlist,
        "allowlist_sha256": {
            path: sha256(files[path]) for path in allowlist
        },
        "secret_pattern_scan": "PASS",
    }
    SUBSET_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "SPACE_CANDIDATE_CHECK "
        + json.dumps(
            {
                "old_paths_subset": True,
                "old_paths": len(protected),
                "candidate_paths": len(files),
                "byte_identical_old_paths": len(protected)
                - len(changed_old_paths),
                "changed_navigation_paths": changed_old_paths,
                "changed_old_evidence_paths": unexpected_changes,
                "allowlist_paths": len(allowlist),
                "secret_scan": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
