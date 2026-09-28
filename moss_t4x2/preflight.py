from __future__ import annotations

import re

from .contract import runtime_contract

QUALIFIED_GPU_PATTERN = re.compile(r"^tesla t4$")
WHITESPACE = re.compile(r"\s+")
VENDOR_PREFIX = re.compile(r"^nvidia\s+")


def normalize_gpu_name(name: str) -> str | None:
    """Return the canonical lowercase Tesla T4 name, or None for anything else."""
    normalized = VENDOR_PREFIX.sub("", WHITESPACE.sub(" ", str(name).strip().lower()))
    return normalized if QUALIFIED_GPU_PATTERN.match(normalized) else None


def evaluate_gpu_inventory(names: list[str]) -> dict[str, object]:
    """Pass only for exactly two Tesla T4 devices; unknown names are never accepted."""
    contract = runtime_contract()
    required = int(contract["required_gpu_count"])
    if len(names) != required:
        raise RuntimeError(
            f"qualified path requires exactly two CUDA GPUs, found {len(names)}: {names}"
        )
    rejected = [name for name in names if normalize_gpu_name(name) is None]
    if rejected:
        raise RuntimeError(
            "qualified path requires two NVIDIA Tesla T4 GPUs; rejected device names: "
            + ", ".join(map(str, rejected))
        )
    return {
        "status": "PASS",
        "gpu_count": len(names),
        "gpu_names": [str(name) for name in names],
        "normalized_gpu_names": [normalize_gpu_name(name) for name in names],
        "required_gpu_pattern": QUALIFIED_GPU_PATTERN.pattern,
        "precision": contract["precision"],
    }