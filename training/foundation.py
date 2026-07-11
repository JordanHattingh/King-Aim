"""Dependency-light primitives shared by King Aim training foundation tools."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, patterns: tuple[str, ...] = ("*",)) -> str:
    digest = hashlib.sha256()
    files = sorted({path for pattern in patterns for path in root.rglob(pattern) if path.is_file()})
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "captured_at_utc": utc_now(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("ultralytics", "torch", "torchvision", "onnx", "onnxruntime")},
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        }
    except ImportError:
        report["torch"] = None
    return report


def write_checksums(root: Path, files: Iterable[Path], destination: Path) -> None:
    rows = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
