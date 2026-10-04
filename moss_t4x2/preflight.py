from __future__ import annotations
from .contract import runtime_contract


def evaluate_gpu_inventory(names: list[str]) -> dict[str, object]:
    c = runtime_contract()
    required = int(c["required_gpu_count"])
    if len(names) != required:
        raise RuntimeError(f"qualified path requires exactly two CUDA GPUs, found {len(names)}")
    if any("Tesla T4" not in name and "T4" not in name for name in names):
        raise RuntimeError(f"qualified path requires two NVIDIA Tesla T4 GPUs, got {names}")
    return {"status": "PASS", "gpu_count": len(names), "gpu_names": names, "precision": c["precision"]}
