#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel

from generate import repository_provenance, weight_map_evidence
from moss_t4x2.contract import runtime_contract
from moss_t4x2.device_map import build_device_map
from moss_t4x2.generation import count_generated_steps, generation_plan, load_processor
from moss_t4x2.metrics import generation_metrics, validate_waveform
from moss_t4x2.model_discovery import discover_model, verify_checkpoint_dtypes, verify_model


def _load_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases manifest must contain a non-empty list")
    seen: set[str] = set()
    normalized = []
    for item in cases:
        case_id = str(item["id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        normalized.append(
            {
                "id": case_id,
                "language": str(item.get("language", "English")),
                "text": str(item["text"]),
                "seed": int(item.get("seed", 42)),
                "max_new_tokens": int(item.get("max_new_tokens", 256)),
            }
        )
    return normalized


def _clear_cuda_peaks() -> None:
    for index in range(2):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-lifecycle MOSS-TTS v1.5 BF16 showcase runner")
    parser.add_argument("--model-root", default="/kaggle/input/models")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-report", required=True)
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = discover_model(Path(args.model_root))
    identity = verify_model(model_dir)
    dtype_evidence = verify_checkpoint_dtypes(model_dir, list(identity["shards"]))
    provenance = repository_provenance(model_dir)

    prepared: dict[str, dict[str, object]] = {}
    print("STAGE=PROCESSOR_SETUP START", flush=True)
    processor_setup_start = time.perf_counter()
    processor = load_processor(model_dir)
    for case in cases:
        case_start = time.perf_counter()
        conversation = [[processor.build_user_message(text=case["text"], language=case["language"])]]
        batch = processor(conversation, mode="generation")
        prepared[str(case["id"])] = {
            "input_ids": batch["input_ids"].cpu(),
            "attention_mask": batch["attention_mask"].cpu(),
            "processor_setup_seconds": time.perf_counter() - case_start,
        }
        del batch, conversation
    if getattr(processor, "audio_tokenizer", None) is not None:
        processor.audio_tokenizer = processor.audio_tokenizer.cpu()
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    processor_setup_total = time.perf_counter() - processor_setup_start
    print(f"STAGE=PROCESSOR_SETUP DONE seconds={processor_setup_total:.3f}", flush=True)

    _clear_cuda_peaks()
    print("STAGE=MODEL_LOAD START", flush=True)
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=build_device_map(),
    )
    shared_backbone_load_seconds = time.perf_counter() - load_start
    print(f"STAGE=MODEL_LOAD DONE seconds={shared_backbone_load_seconds:.3f}", flush=True)
    model.eval()

    tensor_map = weight_map_evidence(model_dir, model)
    if tensor_map["status"] != "PASS":
        raise RuntimeError(
            "loaded module set does not match the checkpoint weight map: "
            f"{tensor_map[unexpected_model_tensors]} / {tensor_map[missing_model_tensors]}"
        )

    cpu_outputs_by_case: dict[str, list[tuple[int, torch.Tensor]]] = {}
    generation_state: dict[str, dict[str, object]] = {}

    for case in cases:
        case_id = str(case["id"])
        print(f"CASE_START={case_id}", flush=True)
        torch.manual_seed(int(case["seed"]))
        torch.cuda.manual_seed_all(int(case["seed"]))
        _clear_cuda_peaks()
        input_ids = prepared[case_id]["input_ids"].to("cuda:0")
        attention_mask = prepared[case_id]["attention_mask"].to("cuda:0")
        torch.cuda.synchronize()
        print(f"STAGE=GENERATE START case={case_id}", flush=True)
        generate_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(case["max_new_tokens"]),
                audio_temperature=1.7,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )
        torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - generate_start
        print(f"STAGE=GENERATE DONE case={case_id} seconds={generate_seconds:.3f}", flush=True)
        gpu_peaks = []
        for index in range(2):
            with torch.cuda.device(index):
                gpu_peaks.append(torch.cuda.max_memory_allocated() / 1024**3)
        cpu_outputs_by_case[case_id] = [(int(start), ids.detach().cpu()) for start, ids in outputs]
        generation_state[case_id] = {
            "generate_seconds": generate_seconds,
            "generated_steps": count_generated_steps(outputs),
            "gpu_peaks_gib": gpu_peaks,
        }
        del outputs, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

    print("STAGE=BACKBONE_RELEASE START", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("STAGE=BACKBONE_RELEASE DONE", flush=True)

    print("STAGE=DECODER_LOAD START", flush=True)
    decoder_load_start = time.perf_counter()
    processor = load_processor(model_dir)
    processor.audio_tokenizer = processor.audio_tokenizer.to("cuda:0")
    shared_decoder_load_seconds = time.perf_counter() - decoder_load_start
    print(f"STAGE=DECODER_LOAD DONE seconds={shared_decoder_load_seconds:.3f}", flush=True)

    reports: dict[str, dict[str, object]] = {}
    for index, case in enumerate(cases):
        case_id = str(case["id"])
        print(f"STAGE=DECODE START case={case_id}", flush=True)
        decode_start = time.perf_counter()
        with torch.inference_mode():
            messages = processor.decode(cpu_outputs_by_case[case_id])
        decode_seconds = time.perf_counter() - decode_start
        print(f"STAGE=DECODE DONE case={case_id} seconds={decode_seconds:.3f}", flush=True)
        audio = messages[0].audio_codes_list[0].detach().float().cpu()
        duration = audio.numel() / float(runtime_contract()["sample_rate_hz"])
        wav_path = output_dir / f"{case_id}.wav"
        torchaudio.save(str(wav_path), audio.unsqueeze(0), runtime_contract()["sample_rate_hz"])

        plan = generation_plan(
            str(case["text"]),
            str(case["language"]),
            seed=int(case["seed"]),
            max_new_tokens=int(case["max_new_tokens"]),
        )
        state = generation_state[case_id]
        report = {
            **plan,
            "case_id": case_id,
            "model_dir": str(model_dir),
            "model_identity": identity,
            "checkpoint_dtype_verification": dtype_evidence,
            "weight_map_match": tensor_map,
            "model_patch": False,
            "architecture_modified": False,
            "provenance": provenance,
            "sample_rate": runtime_contract()["sample_rate_hz"],
            "num_samples": int(audio.numel()),
            "wav_path": str(wav_path),
            "signal": validate_waveform(audio.tolist()),
            "shared_lifecycle": True,
            "shared_backbone_load_seconds": shared_backbone_load_seconds,
            "shared_decoder_load_seconds": shared_decoder_load_seconds,
            **generation_metrics(
                processor_setup_seconds=float(prepared[case_id]["processor_setup_seconds"]),
                model_load_seconds=shared_backbone_load_seconds if index == 0 else 0.0,
                generate_seconds=float(state["generate_seconds"]),
                decode_seconds=decode_seconds,
                audio_duration_seconds=duration,
                generated_steps=int(state["generated_steps"]),
                gpu_peaks_gib=list(state["gpu_peaks_gib"]),
            ),
        }
        report_path = output_dir / f"{case_id}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if report["signal"]["status"] != "PASS":
            raise RuntimeError(f"invalid waveform for {case_id}")
        reports[case_id] = report
        print(
            f"CASE_DONE={case_id} duration={duration:.2f}s "
            f"generation_rtf={report["generation_rtf"]:.3f} decode_rtf={report["decode_rtf"]:.3f}",
            flush=True,
        )
        del messages, audio

    summary = {
        "status": "PASS",
        "case_count": len(cases),
        "case_ids": [str(case["id"]) for case in cases],
        "shared_backbone_load_count": 1,
        "shared_decoder_load_count": 1,
        "shared_backbone_load_seconds": shared_backbone_load_seconds,
        "shared_decoder_load_seconds": shared_decoder_load_seconds,
        "reports": {case_id: str(output_dir / f"{case_id}.json") for case_id in reports},
    }
    Path(args.summary_report).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("MOSS_TTS_BATCH_GENERATION=PASS")


if __name__ == "__main__":
    sys.exit(main())
