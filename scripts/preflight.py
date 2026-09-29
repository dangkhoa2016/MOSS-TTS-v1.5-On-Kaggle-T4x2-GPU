#!/usr/bin/env python3
"""Machine-readable GPU preflight for the qualified Kaggle T4 x2 path."""
from __future__ import annotations

import argparse
import json
import sys

import torch

from moss_t4x2.preflight import evaluate_gpu_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle T4 x2 preflight")
    parser.add_argument("--report", default=None, help="optional path for the JSON preflight report")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select Kaggle GPU T4 x2")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    report = evaluate_gpu_inventory(names)

    if args.report:
        from pathlib import Path

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["status"] == "PASS"
    assert report["gpu_count"] == 2
    print("GPU_T4X2=PASS")


if __name__ == "__main__":
    sys.exit(main())