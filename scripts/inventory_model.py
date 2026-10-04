#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from moss_t4x2.inventory import inventory_model
from moss_t4x2.model_discovery import discover_model


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--root", default="/kaggle/input/models")
    args=p.parse_args(); model=discover_model(Path(args.root)); report=inventory_model(model)
    print(json.dumps(report, indent=2)); print("MODEL_ORIGINAL_SAFETENSORS=PASS")

if __name__ == "__main__": main()
