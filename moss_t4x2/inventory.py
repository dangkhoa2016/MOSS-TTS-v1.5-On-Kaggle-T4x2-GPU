from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .contract import runtime_contract
from .model_discovery import REQUIRED_FILES, verify_checkpoint_dtypes, verify_model

REFERENCE_SCHEMA = "moss-t4x2/checkpoint-sha256/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ABSENT_REFERENCE = {
    "status": "ABSENT",
    "gating": False,
    "note": "no sha256 reference manifest supplied; originality is not cryptographically proven",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_reference_manifest(path: Path) -> dict[str, object]:
    """Load and validate a reference manifest of canonical checkpoint digests."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid sha256 reference manifest {path.name}: {error}") from error
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise RuntimeError(f"sha256 reference manifest {path.name} must declare schema={REFERENCE_SCHEMA}")
    confirmation = manifest.get("confirmation")
    if confirmation not in {"pending", "confirmed"}:
        raise RuntimeError(f"sha256 reference manifest {path.name} needs confirmation pending|confirmed")
    expected = manifest.get("expected_sha256", {})
    if not isinstance(expected, dict):
        raise RuntimeError(f"sha256 reference manifest {path.name} expected_sha256 must be an object")
    invalid = [name for name, digest in expected.items() if not isinstance(digest, str) or not _SHA256_RE.match(digest)]
    if invalid:
        raise RuntimeError(f"sha256 reference manifest {path.name} has malformed sha256 digest entries: {invalid}")
    return {
        "schema": REFERENCE_SCHEMA,
        "confirmation": confirmation,
        "model_id": manifest.get("model_id"),
        "kaggle_mirror": manifest.get("kaggle_mirror"),
        "note": manifest.get("note", ""),
        "expected_sha256": expected,
        "development_measured_sha256": manifest.get("development_measured_sha256", {}),
    }


def compare_sha256_reference(
    model_dir: Path, files: dict[str, dict[str, object]], reference_path: Path | None
) -> dict[str, object]:
    """Compare attached checkpoint files against canonical digests when they are confirmed."""
    if reference_path is None or not Path(reference_path).is_file():
        return dict(_ABSENT_REFERENCE)
    manifest = sha256_reference_manifest(reference_path)
    expected = manifest["expected_sha256"]
    if not expected:
        return {
            "status": "UNVERIFIED",
            "gating": False,
            "manifest": Path(reference_path).name,
            "manifest_confirmation": manifest["confirmation"],
            "compared_files": 0,
            "matched": [],
            "mismatched": [],
            "missing": [],
            "note": manifest["note"] or "expected_sha256 is empty; populate and confirm before release",
        }
    matched: list[str] = []
    mismatched: list[dict[str, str]] = []
    missing: list[str] = []
    for name, digest in sorted(expected.items()):
        if name not in files:
            missing.append(name)
        elif files[name]["sha256"] == digest:
            matched.append(name)
        else:
            mismatched.append({"file": name, "expected": digest, "actual": str(files[name]["sha256"])})
    if missing:
        raise RuntimeError(f"missing expected files for sha256 comparison: {missing}")
    if mismatched:
        detail = ", ".join(item["file"] for item in mismatched)
        raise RuntimeError(f"sha256 mismatch against the canonical checkpoint manifest: {detail}")
    return {
        "status": "PASS",
        "gating": True,
        "manifest": Path(reference_path).name,
        "manifest_confirmation": manifest["confirmation"],
        "compared_files": len(expected),
        "matched": matched,
        "mismatched": [],
        "missing": [],
    }


def inventory_model(model_dir: Path, reference_path: Path | None = None) -> dict[str, object]:
    """Report filesystem inventory, checkpoint identity and dtype evidence separately."""
    model_dir = Path(model_dir)
    contract = runtime_contract()
    identity = verify_model(model_dir)

    gguf_files = sorted(path.name for path in model_dir.glob("*.gguf"))
    if gguf_files:
        raise RuntimeError(f"GGUF artifacts are out of contract for the qualified path: {gguf_files}")

    dtype_evidence = verify_checkpoint_dtypes(model_dir, list(identity["shards"]))

    targets = [model_dir / name for name in REQUIRED_FILES]
    targets.append(model_dir / "processor_config.json")
    targets.extend(model_dir / name for name in identity["shards"])
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in targets
        if path.is_file()
    }

    return {
        "status": "PASS",
        "identity": {
            "model_id": identity["model_id"],
            "kaggle_model": identity["kaggle_model"],
            "architecture": identity["architecture"],
            "model_type": identity["model_type"],
            "declared_dtype": identity["declared_dtype"],
            "tensor_count": identity["tensor_count"],
            "shards": identity["shards"],
        },
        "format": {
            "safetensors": True,
            "gguf": False,
            "shard_suffix": ".safetensors",
        },
        "quantization": contract["quantization"],
        "model_patch": False,
        "shard_count": len(identity["shards"]),
        "checkpoint_dtype_verification": dtype_evidence,
        "checkpoint_sha256_reference": compare_sha256_reference(model_dir, files, reference_path),
        "files": files,
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "runtime_contract": {
            "precision": contract["precision"],
            "batch_size": contract["batch_size"],
            "sample_rate_hz": contract["sample_rate_hz"],
            "required_gpu_count": contract["required_gpu_count"],
        },
    }