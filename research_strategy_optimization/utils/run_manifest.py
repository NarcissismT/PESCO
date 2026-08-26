"""Machine-readable provenance manifests for reproducible experiment runs.

The Tier-1 artifacts are deliberately diagnostic rather than paper-level claims,
but they still need an auditable execution boundary.  This module keeps that
boundary independent from any particular runner: callers can record the command,
source files, data files, random seeds, runtime package versions, and (when a
checkpoint is supplied) complete checkpoint/weight/tokenizer digests.

The helpers are standard-library only.  Optional packages such as NumPy, PyTorch,
Transformers, and Safetensors are queried through ``importlib.metadata`` and are
reported as unavailable instead of making a core/reference run fail.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_MANIFEST_SCHEMA = "pesco_run_manifest_v0.1"
_DEFAULT_CHUNK_SIZE = 1024 * 1024


def _sha256_stream(stream: Any, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def digest_file(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    source = Path(path)
    with source.open("rb") as handle:
        return _sha256_stream(handle)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _relative_or_absolute(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return str(resolved)


def _file_entries(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    role: str = "input",
) -> list[dict[str, Any]]:
    """Return deterministic file entries for files and recursively supplied dirs."""

    root_path = Path(root).resolve() if root is not None else None
    expanded: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute() and root_path is not None:
            path = root_path / path
        if not path.exists():
            continue
        if path.is_dir():
            children = sorted(child for child in path.rglob("*") if child.is_file())
        else:
            children = [path]
        for child in children:
            key = _relative_or_absolute(child, root_path)
            expanded[key] = child
    entries: list[dict[str, Any]] = []
    for key in sorted(expanded):
        child = expanded[key]
        entries.append(
            {
                "path": key,
                "size_bytes": int(child.stat().st_size),
                "sha256": digest_file(child),
                "role": role,
            }
        )
    return entries


def digest_file_entries(entries: Sequence[Mapping[str, Any]]) -> str | None:
    """Digest a deterministic list of file metadata, or ``None`` when empty."""

    if not entries:
        return None
    return _canonical_digest([dict(entry) for entry in entries])


def digest_paths(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    role: str = "input",
) -> dict[str, Any]:
    """Hash supplied files/directories and return entries plus aggregate digest."""

    entries = _file_entries(paths, root=root, role=role)
    return {
        "available": bool(entries),
        "file_count": len(entries),
        "files": entries,
        "digest": digest_file_entries(entries),
    }


_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}
_TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.txt",
    "vocab.json",
}
_CONFIG_NAMES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
}


def checkpoint_inventory(path: str | Path | None, *, root: str | Path | None = None) -> dict[str, Any]:
    """Inventory a local checkpoint, including complete weights and tokenizer hashes.

    Missing checkpoints are represented explicitly.  A missing checkpoint is not
    silently converted to an empty digest, which prevents a manifest from looking
    reproducible when the external model was unavailable.
    """

    if path is None:
        return {
            "supplied": False,
            "available": False,
            "path": None,
            "reason": "no_checkpoint_supplied",
            "full_checkpoint_digest": None,
            "weights_digest": None,
            "full_weights_digest": None,
            "weights_file_count": 0,
            "tokenizer_digest": None,
            "full_tokenizer_digest": None,
            "tokenizer_file_count": 0,
            "config_digest": None,
            "file_count": 0,
            "files": [],
        }
    checkpoint = Path(path)
    if not checkpoint.is_absolute() and root is not None:
        checkpoint = Path(root).resolve() / checkpoint
    display_path = _relative_or_absolute(checkpoint, Path(root) if root is not None else None)
    if not checkpoint.exists():
        return {
            "supplied": True,
            "available": False,
            "path": display_path,
            "reason": "checkpoint_path_missing",
            "full_checkpoint_digest": None,
            "weights_digest": None,
            "full_weights_digest": None,
            "weights_file_count": 0,
            "tokenizer_digest": None,
            "full_tokenizer_digest": None,
            "tokenizer_file_count": 0,
            "config_digest": None,
            "file_count": 0,
            "files": [],
        }

    all_entries = _file_entries([checkpoint], root=root, role="checkpoint")
    weight_paths = [
        item for item in all_entries if Path(str(item["path"])).suffix.lower() in _WEIGHT_SUFFIXES
    ]
    tokenizer_paths = []
    for item in all_entries:
        name = Path(str(item["path"])).name.lower()
        if name in _TOKENIZER_NAMES or name.endswith(".model"):
            tokenizer_paths.append(item)
    config_paths = [
        item for item in all_entries if Path(str(item["path"])).name.lower() in _CONFIG_NAMES
    ]
    weights_digest = digest_file_entries(weight_paths)
    tokenizer_digest = digest_file_entries(tokenizer_paths)
    return {
        "supplied": True,
        "available": True,
        "path": display_path,
        "reason": None,
        "full_checkpoint_digest": digest_file_entries(all_entries),
        "weights_digest": weights_digest,
        "full_weights_digest": weights_digest,
        "weights_file_count": len(weight_paths),
        "tokenizer_digest": tokenizer_digest,
        "full_tokenizer_digest": tokenizer_digest,
        "tokenizer_file_count": len(tokenizer_paths),
        "config_digest": digest_file_entries(config_paths),
        "file_count": len(all_entries),
        "files": all_entries,
    }


def _run_git(repo_root: Path, *args: str, timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_inventory(repo_root: str | Path) -> dict[str, Any]:
    """Return commit and working-tree provenance without requiring GitPython."""

    root = Path(repo_root).resolve()
    sha = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "branch", "--show-current")
    # This repository contains many generated JSONL artifacts; a full porcelain
    # scan can take a few seconds on the shared filesystem.  Give status enough
    # time to finish so a slow scan is never misreported as a clean tree.
    status = _run_git(root, "status", "--porcelain", timeout=60.0)
    return {
        "repository": str(root),
        "sha": sha,
        "branch": branch or None,
        "dirty": (bool(status) if status is not None else None),
        "status_available": status is not None,
        "status_digest": _canonical_digest(status.splitlines()) if status else None,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def runtime_inventory() -> dict[str, Any]:
    """Capture interpreter/platform and optional scientific package versions."""

    distributions = (
        "numpy",
        "torch",
        "transformers",
        "safetensors",
        "accelerate",
        "matplotlib",
    )
    packages = {name: _package_version(name) for name in distributions}
    package_versions_digest = _canonical_digest(packages)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        # Explicit aliases keep the required versions easy to consume without
        # knowing the optional-package map's distribution names.
        "numpy_version": packages["numpy"],
        "pytorch_version": packages["torch"],
        "dependency_versions_digest": package_versions_digest,
    }


def command_inventory(command: Sequence[str] | None = None, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Record argv losslessly and provide a shell-rendered convenience form."""

    argv = list(command) if command is not None else list(sys.argv)
    return {
        "argv": [str(item) for item in argv],
        "shell": shlex.join(str(item) for item in argv),
        "cwd": str(Path(cwd or os.getcwd()).resolve()),
    }


def seed_inventory(seeds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize seed records while preserving semantic categories."""

    normalized = {str(key): value for key, value in (seeds or {}).items()}
    # A canonical ``all`` field makes downstream audits independent of runner
    # naming, while retaining the original categories (training/exploration/etc.).
    values: list[int] = []
    for key, value in normalized.items():
        if key == "all":
            continue
        if isinstance(value, (list, tuple)):
            values.extend(int(item) for item in value)
        elif isinstance(value, int) and not isinstance(value, bool):
            values.append(int(value))
    normalized["all"] = sorted(set(values))
    return normalized


def build_run_manifest(
    *,
    experiment: str,
    repo_root: str | Path,
    command: Sequence[str] | None = None,
    runner_paths: Iterable[str | Path] = (),
    data_paths: Iterable[str | Path] = (),
    seeds: Mapping[str, Any] | None = None,
    checkpoint: str | Path | None = None,
    status: str = "completed",
    diagnostics: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a complete run manifest suitable for JSON serialization.

    ``runner_paths`` and ``data_paths`` may contain directories; all regular files
    beneath them are hashed.  ``diagnostics`` is intentionally namespaced so callers
    can retain experiment-specific values without changing the core provenance
    schema.
    """

    root = Path(repo_root).resolve()
    source = digest_paths(runner_paths, root=root, role="source")
    data = digest_paths(data_paths, root=root, role="data")
    dependency_specs = digest_paths(
        [
            root / "requirements.txt",
            root / "pyproject.toml",
            root / "environment.yml",
            root / "environment.yaml",
        ],
        root=root,
        role="dependency_spec",
    )
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "experiment": str(experiment),
        "status": str(status),
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "git": git_inventory(root),
        "command": command_inventory(command, cwd=root),
        "runtime": runtime_inventory(),
        "dependencies": dependency_specs,
        "seeds": seed_inventory(seeds),
        "data": data,
        "source": source,
        "checkpoint": checkpoint_inventory(checkpoint, root=root),
        "diagnostics": dict(diagnostics or {}),
    }
    # Keep both structured inventories and a small flat index for shell/CI tools
    # that need the required P0 fields without schema-specific traversal.
    payload["git_sha"] = payload["git"].get("sha")
    payload["command_line"] = payload["command"].get("shell")
    payload["python_version"] = payload["runtime"].get("python_version")
    payload["numpy_version"] = payload["runtime"].get("numpy_version")
    payload["pytorch_version"] = payload["runtime"].get("pytorch_version")
    payload["training_seed"] = payload["seeds"].get("training")
    payload["data_digest"] = payload["data"].get("digest")
    payload["source_digest"] = payload["source"].get("digest")
    payload["checkpoint_full_digest"] = payload["checkpoint"].get("full_checkpoint_digest")
    payload["checkpoint_weights_digest"] = payload["checkpoint"].get("full_weights_digest")
    payload["checkpoint_tokenizer_digest"] = payload["checkpoint"].get("full_tokenizer_digest")
    payload["dependency_versions_digest"] = payload["runtime"].get("dependency_versions_digest")
    payload["dependency_spec_digest"] = payload["dependencies"].get("digest")
    payload["manifest_digest"] = _canonical_digest(payload)
    return payload


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Recompute the canonical digest for an existing manifest mapping."""

    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    return _canonical_digest(payload)


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Write a manifest atomically enough for ordinary local experiment use."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(manifest), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)
    return dict(manifest)


__all__ = [
    "RUN_MANIFEST_SCHEMA",
    "build_run_manifest",
    "checkpoint_inventory",
    "command_inventory",
    "digest_file",
    "digest_file_entries",
    "digest_paths",
    "git_inventory",
    "runtime_inventory",
    "seed_inventory",
    "manifest_digest",
    "write_run_manifest",
]
