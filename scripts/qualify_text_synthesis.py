#!/usr/bin/env python3
"""Technical execution qualification for English, Vietnamese and code-switch synthesis."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CASES = [
    ("en", "English", "Today we are testing clear speech on two Tesla T4 GPUs."),
    ("vi", "Vietnamese", "Xin chào. Đây là bài kiểm tra tổng hợp giọng nói tiếng Việt trên hai GPU Tesla T4."),
    (
        "codeswitch",
        "Vietnamese",
        "Xin chào, đây là Moss TTS on Kaggle, và this sentence switches between Vietnamese and English.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical text-synthesis execution qualification")
    parser.add_argument("--model-root", default="/kaggle/input/models")
    parser.add_argument("--output-dir", default="results/text-synthesis")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id, language, text in CASES:
        wav = out / f"{case_id}.wav"
        report_path = out / f"{case_id}.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/generate.py",
                "--model-root",
                args.model_root,
                "--text",
                text,
                "--language",
                language,
                "--output",
                str(wav),
                "--report",
                str(report_path),
            ],
            check=True,
        )
        report = json.loads(report_path.read_text())
        assert wav.is_file() and wav.stat().st_size > 0, f"missing audio for {case_id}"
        assert report["signal"]["status"] == "PASS", case_id
        assert report["model_identity"]["status"] == "PASS", case_id
        rows.append(
            {
                "case": case_id,
                "language": language,
                "wav_bytes": wav.stat().st_size,
                "audio_duration_seconds": report["audio_duration_seconds"],
                "generation_rtf": report["generation_rtf"],
                "decode_rtf": report["decode_rtf"],
                "generated_steps": report["generated_steps"],
                "declared_dtype": report["model_identity"]["declared_dtype"],
            }
        )

    summary = {
        "scope": "technical_execution_only",
        "cases": rows,
        "human_listening_review": "PENDING",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    assert {row["case"] for row in rows} == {"en", "vi", "codeswitch"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("TEXT_SYNTHESIS_TECHNICAL_QUALIFICATION=PASS")
    print("HUMAN_LISTENING_REVIEW=PENDING")


if __name__ == "__main__":
    sys.exit(main())