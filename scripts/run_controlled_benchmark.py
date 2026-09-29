#!/usr/bin/env python3
"""Cold-process controlled generation benchmark.

Each run is a fresh process that reloads the checkpoint, so these numbers describe a
cold-process measurement and must not be reported as warm-resident serving throughput.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

STAGE_KEYS = (
    "processor_setup_seconds",
    "model_load_seconds",
    "generate_seconds",
    "decode_seconds",
    "total_seconds",
    "audio_duration_seconds",
    "generation_rtf",
    "decode_rtf",
    "generated_steps",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-process controlled generation benchmark")
    parser.add_argument("--model-root", default="/kaggle/input/models")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", default="results/benchmark")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = "This controlled benchmark measures original BF16 Moss TTS generation on two Tesla T4 GPUs."
    rows = []
    wall_start = time.perf_counter()
    for index in range(args.runs):
        wav = out / f"run-{index + 1}.wav"
        report_path = out / f"run-{index + 1}.json"
        run_start = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                "scripts/generate.py",
                "--model-root",
                args.model_root,
                "--text",
                text,
                "--language",
                "English",
                "--output",
                str(wav),
                "--report",
                str(report_path),
            ],
            check=True,
        )
        wall_seconds = time.perf_counter() - run_start
        report = json.loads(report_path.read_text())
        assert report["signal"]["status"] == "PASS"
        row = {key: report[key] for key in STAGE_KEYS}
        row["run"] = index + 1
        row["process_wall_seconds"] = wall_seconds
        row["gpu0_peak_alloc_gib"] = report["gpu0_peak_alloc_gib"]
        row["gpu1_peak_alloc_gib"] = report["gpu1_peak_alloc_gib"]
        row["repository_commit_sha"] = report["provenance"]["repository_commit_sha"]
        rows.append(row)
    benchmark_wall_seconds = time.perf_counter() - wall_start

    summary = {
        "methodology": "cold_process_per_run",
        "model_residency": "reloaded_per_run",
        "steady_state_throughput": False,
        "runs": args.runs,
        "median_generation_rtf": statistics.median(row["generation_rtf"] for row in rows),
        "median_decode_rtf": statistics.median(row["decode_rtf"] for row in rows),
        "median_generate_seconds": statistics.median(row["generate_seconds"] for row in rows),
        "median_model_load_seconds": statistics.median(row["model_load_seconds"] for row in rows),
        "median_decode_seconds": statistics.median(row["decode_seconds"] for row in rows),
        "median_audio_duration_seconds": statistics.median(row["audio_duration_seconds"] for row in rows),
        "median_generated_steps": statistics.median(row["generated_steps"] for row in rows),
        "benchmark_wall_seconds": benchmark_wall_seconds,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert summary["steady_state_throughput"] is False
    assert len(rows) == args.runs
    print("COLD_PROCESS_CONTROLLED_BENCHMARK=PASS")


if __name__ == "__main__":
    sys.exit(main())