from __future__ import annotations

import json
from pathlib import Path

from .contract import runtime_contract

_REQUIRED = ("config.json", "model.safetensors.index.json")


def _looks_like_model(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in _REQUIRED)


def discover_model(root: Path) -> Path:
    root = Path(root)
    if _looks_like_model(root):
        return root
    candidates = sorted({p.parent for p in root.rglob("model.safetensors.index.json") if _looks_like_model(p.parent)})
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
    model_dir = Path(model_dir)
    missing = [name for name in _REQUIRED if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required model files: {missing}")
    cfg = json.loads((model_dir / "config.json").read_text())
    contract = runtime_contract()
    architectures = cfg.get("architectures") or []
    if contract["architecture"] not in architectures:
        raise RuntimeError(f"expected {contract['architecture']} in config architectures, got {architectures}")
    if cfg.get("model_type") != contract["model_type"]:
        raise RuntimeError(f"expected model_type={contract['model_type']}, got {cfg.get('model_type')}")
    dtype = cfg.get("torch_dtype") or cfg.get("dtype")
    if dtype not in {None, "bfloat16"}:
        raise RuntimeError(f"expected BF16 checkpoint config, got dtype={dtype}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set((index.get("weight_map") or {}).values()))
    missing_shards = [name for name in shard_names if not (model_dir / name).is_file()]
    if missing_shards:
        raise RuntimeError(f"missing safetensor shards: {missing_shards}")
    return {
        "path": str(model_dir),
        "architecture": contract["architecture"],
        "model_type": contract["model_type"],
        "dtype": dtype or "bfloat16",
        "shards": shard_names,
    }
