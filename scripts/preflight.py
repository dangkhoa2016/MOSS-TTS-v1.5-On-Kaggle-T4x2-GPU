#!/usr/bin/env python3
import json, torch
from moss_t4x2.preflight import evaluate_gpu_inventory


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select Kaggle GPU T4 x2")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    report = evaluate_gpu_inventory(names)
    print(json.dumps(report, indent=2))
    print("GPU_T4X2=PASS")

if __name__ == "__main__": main()
