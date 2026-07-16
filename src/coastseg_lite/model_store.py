"""Pinned model download and integrity verification."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any


def bundled_manifest_path() -> Path:
    return Path(str(files("coastseg_lite").joinpath("model_manifest.json")))


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or bundled_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "coastseg-lite.model/v1":
        raise ValueError(f"Unsupported model manifest: {manifest_path}")
    if path is None:
        architecture = manifest["architecture"]
        config_path = bundled_manifest_path().with_name(architecture["bundled_config"])
        actual = sha256_file(config_path)
        if actual != architecture["bundled_config_sha256"]:
            raise ValueError(f"Bundled architecture checksum mismatch: {config_path}")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_url(manifest: dict[str, Any], name: str) -> str:
    source = manifest["source"]
    return (
        f"https://huggingface.co/{source['repository']}/resolve/"
        f"{source['revision']}/{name}"
    )


def _verify_file(path: Path, record: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Model file size mismatch: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ValueError(
            f"Model file checksum mismatch: {path}; "
            f"expected {record['sha256']}, got {actual}"
        )


def verify_model_dir(
    model_dir: Path, manifest_path: Path | None = None
) -> dict[str, str]:
    manifest = load_manifest(manifest_path)
    verified: dict[str, str] = {}
    for record in manifest["files"]:
        path = model_dir / record["name"]
        _verify_file(path, record)
        verified[record["name"]] = record["sha256"]
    return verified


def _download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "coastseg-lite/0.1"})
    temporary = output.with_name(f"{output.name}.part")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while block := response.read(1024 * 1024):
                stream.write(block)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_model(
    model_dir: Path,
    manifest_path: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, str]:
    manifest = load_manifest(manifest_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    for record in manifest["files"]:
        path = model_dir / record["name"]
        if path.exists():
            try:
                _verify_file(path, record)
                continue
            except ValueError:
                if not force:
                    raise
                path.unlink()
        _download(_source_url(manifest, record["name"]), path)
        _verify_file(path, record)
    return verify_model_dir(model_dir, manifest_path)


def best_model_path(model_dir: Path, manifest_path: Path | None = None) -> Path:
    manifest = load_manifest(manifest_path)
    verify_model_dir(model_dir, manifest_path)
    path = model_dir / manifest["best_model"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
