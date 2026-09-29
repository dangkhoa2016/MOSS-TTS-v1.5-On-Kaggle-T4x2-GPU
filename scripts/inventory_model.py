#!/usr/bin/env python3
"""Report filesystem inventory, checkpoint identity and dtype evidence for a checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from moss_t4x2.inventory import inventory_model
from moss_t4x2.model_discovery import discover_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = REPOSITORY_ROOT / "references" / "checkpoint-sha256.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS v1.5 checkpoint inventory")
    parser.add_argument("--root", default="/kaggle/input/models")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model_dir = discover_model(Path(args.root))
    reference = Path(args.reference) if args.reference else None
    report = inventory_model(model_dir, reference_path=reference)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    assert report["status"] == "PASS"
    assert report["identity"]["declared_dtype"] == "bfloat16"
    assert report["checkpoint_dtype_verification"]["status"] == "PASS"
    assert report["format"]["gguf"] is False
    assert report["quantization"] == "none"
    assert report["checkpoint_sha256_reference"]["status"] in {"PASS", "UNVERIFIED"}
    print("MODEL_IDENTITY_AND_SAFETENSORS=PASS")


if __name__ == "__main__":
    sys.exit(main())