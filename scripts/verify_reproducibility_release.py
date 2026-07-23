#!/usr/bin/env python3
"""Verify the separately published E5 reproducibility release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "results"
    / "resubmission"
    / "v7_cross_instance"
    / "archive_manifest_v8.1.0.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def verify_checkpoint_payload(archive: tarfile.TarFile) -> None:
    files = {
        member.name: archive.extractfile(member).read()
        for member in archive.getmembers()
        if member.isfile()
    }
    metadata_names = sorted(
        name for name in files if name.endswith(".pt.complete.json")
    )
    object_names = sorted(
        name
        for name in files
        if name.startswith("objects/") and name.endswith(".pt")
    )
    if len(metadata_names) != 50 or len(object_names) != 50:
        raise ValueError(
            "checkpoint archive must contain 50 metadata files and 50 objects"
        )

    for metadata_name in metadata_names:
        metadata = json.loads(files[metadata_name])
        object_name = metadata["object_path"]
        if object_name not in files:
            raise ValueError(f"{metadata_name}: missing {object_name}")
        payload = files[object_name]
        if len(payload) != metadata["checkpoint_size_bytes"]:
            raise ValueError(f"{metadata_name}: checkpoint size mismatch")
        if hashlib.sha256(payload).hexdigest() != metadata["checkpoint_sha256"]:
            raise ValueError(f"{metadata_name}: checkpoint SHA-256 mismatch")


def verify_asset(asset_dir: Path, asset: dict[str, object]) -> None:
    path = asset_dir / str(asset["name"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(asset["size_bytes"]):
        raise ValueError(f"{path.name}: byte-size mismatch")
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"{path.name}: archive SHA-256 mismatch")

    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        unsafe = [member.name for member in members if not safe_member_name(member.name)]
        if unsafe:
            raise ValueError(f"{path.name}: unsafe archive path {unsafe[0]!r}")
        file_count = sum(member.isfile() for member in members)
        if file_count != int(asset["file_count"]):
            raise ValueError(
                f"{path.name}: expected {asset['file_count']} files, got {file_count}"
            )
        if "checkpoints" in path.name:
            verify_checkpoint_payload(archive)

    print(f"OK {path.name}: {asset['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        verify_asset(args.asset_dir, asset)
    print(f"Verified {len(manifest['assets'])} release assets.")


if __name__ == "__main__":
    main()
