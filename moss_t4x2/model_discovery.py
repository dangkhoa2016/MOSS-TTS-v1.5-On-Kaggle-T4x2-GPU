from __future__ import annotations

import json
from pathlib import Path

from .contract import runtime_contract

REQUIRED_FILES = ("config.json", "model.safetensors.index.json")
MAX_HEADER_BYTES = 64 * 1024 * 1024
_FLOAT_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}
_QUALIFIED_DTYPE = "bfloat16"


def _looks_like_model(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"expected a JSON object in {path.name}")
    return data


def _declared_dtype(config: dict[str, object]) -> str | None:
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """Read only the safetensors JSON header; tensor payloads are never materialized."""
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"unreadable safetensors header in {path.name}")
        header_length = int.from_bytes(raw_length, "little", signed=False)
        if header_length <= 0 or header_length > MAX_HEADER_BYTES:
            raise RuntimeError(f"implausible safetensors header length in {path.name}: {header_length}")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise RuntimeError(f"truncated safetensors header in {path.name}")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable safetensors header in {path.name}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"unexpected safetensors header structure in {path.name}")
    return {key: value for key, value in header.items() if key != "__metadata__"}


def verify_checkpoint_dtypes(model_dir: Path, shards: list[str]) -> dict[str, object]:
    """Verify stored tensor dtypes from shard headers without loading any weights."""
    model_dir = Path(model_dir)
    floating: dict[str, str] = {}
    non_floating = 0
    for shard in shards:
        for name, descriptor in _read_safetensors_header(model_dir / shard).items():
            dtype = descriptor.get("dtype") if isinstance(descriptor, dict) else None
            if dtype in _FLOAT_DTYPES:
                floating[f"{shard}:{name}"] = _FLOAT_DTYPES[str(dtype)]
            else:
                non_floating += 1
    if not floating:
        raise RuntimeError("no floating-point tensors found in the safetensors headers")
    offenders = sorted({value for value in floating.values() if value != _QUALIFIED_DTYPE})
    if offenders:
        raise RuntimeError(
            "checkpoint contains non-BF16 floating tensors: "
            + ", ".join(offenders)
            + f" (qualified path requires {_QUALIFIED_DTYPE} only)"
        )
    return {
        "status": "PASS",
        "method": "safetensors_header_metadata",
        "weights_materialized": False,
        "floating_dtype": _QUALIFIED_DTYPE,
        "verified_tensor_count": len(floating) + non_floating,
        "floating_tensor_count": len(floating),
        "non_floating_tensor_count": non_floating,
        "non_bfloat16_floating_tensors": [],
        "shards_inspected": list(shards),
    }


def discover_candidates(root: Path) -> list[Path]:
    root = Path(root)
    candidates = sorted({p.parent for p in root.rglob("model.safetensors.index.json") if _looks_like_model(p.parent)})
    return candidates


def count_model_candidates(root: Path) -> list[str]:
    return [str(candidate.relative_to(Path(root))) for candidate in discover_candidates(Path(root))]


def discover_model(root: Path) -> Path:
    """Return the single MOSS-TTS-v1.5 checkpoint under root, verified before acceptance."""
    root = Path(root)
    if _looks_like_model(root):
        verify_model(root)
        return root
    candidates = discover_candidates(root)
    if not candidates:
        raise RuntimeError(f"no MOSS-TTS model candidate found under {root}")
    valid: list[Path] = []
    for candidate in candidates:
        try:
            verify_model(candidate)
        except RuntimeError:
            continue
        valid.append(candidate)
    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise RuntimeError(f"no valid MOSS-TTS-v1.5 candidate found under {root}")
    raise RuntimeError("ambiguous MOSS-TTS-v1.5 model candidates: " + ", ".join(map(str, valid)))


def verify_model(model_dir: Path) -> dict[str, object]:
    """Apply the full identity contract; never infer a missing property."""
    model_dir = Path(model_dir)
    contract = runtime_contract()

    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required model files: {missing}")

    config = _read_json(model_dir / "config.json")
    architectures = config.get("architectures") or []
    if contract["architecture"] not in architectures:
        raise RuntimeError(f"expected {contract['architecture']} in config architectures, got {architectures}")
    if config.get("model_type") != contract["model_type"]:
        raise RuntimeError(f"expected model_type={contract['model_type']}, got {config.get('model_type')}")

    declared_dtype = _declared_dtype(config)
    if declared_dtype != contract["precision"]:
        raise RuntimeError(
            f"expected declared dtype={contract['precision']}, got {declared_dtype!r}; "
            "a missing dtype is not inferred as BF16"
        )

    index = _read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model.safetensors.index.json must declare a non-empty weight_map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names):
        raise RuntimeError(f"weight_map must reference .safetensors shards, got {shard_names}")
    missing_shards = [name for name in shard_names if not (model_dir / name).is_file()]
    if missing_shards:
        raise RuntimeError(f"missing safetensor shards: {missing_shards}")

    declared_quantization = config.get("quantization_config")
    if declared_quantization not in (None, False):
        raise RuntimeError(f"quantized checkpoints are out of contract: {declared_quantization}")

    return {
        "status": "PASS",
        "path": str(model_dir),
        "model_id": contract["model_id"],
        "kaggle_model": contract["kaggle_model"],
        "architecture": contract["architecture"],
        "model_type": contract["model_type"],
        "declared_dtype": declared_dtype,
        "quantization": contract["quantization"],
        "tensor_count": len(weight_map),
        "shards": shard_names,
    }