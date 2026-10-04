from __future__ import annotations
import hashlib, json
from pathlib import Path
from .model_discovery import verify_model


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory_model(model_dir: Path) -> dict[str, object]:
    model_dir = Path(model_dir)
    identity = verify_model(model_dir)
    targets = [model_dir / "config.json", model_dir / "model.safetensors.index.json"] + [model_dir / s for s in identity["shards"]]
    files = {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in targets}
    return {"status": "PASS", "identity": identity, "files": files, "total_bytes": sum(v["bytes"] for v in files.values())}
