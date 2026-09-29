#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel

from moss_t4x2.contract import runtime_contract
from moss_t4x2.device_map import build_device_map
from moss_t4x2.generation import count_generated_steps, generation_plan, load_processor
from moss_t4x2.metrics import generation_metrics, validate_waveform
from moss_t4x2.model_discovery import discover_model, verify_checkpoint_dtypes, verify_model

TRACKED_PACKAGES = ("torch", "torchaudio", "transformers", "accelerate", "safetensors", "huggingface-hub")


def repository_provenance(model_dir: Path) -> dict[str, object]:
    """Capture which source revision and runtime produced this report."""
    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    upstream_lock = Path(__file__).resolve().parents[1] / "references" / "upstream.lock"
    upstream = {}
    if upstream_lock.is_file():
        for line in upstream_lock.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                upstream[key.strip()] = value.strip()
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "repository_commit_sha": _git("rev-parse", "HEAD"),
        "repository_ref": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "upstream_repository": upstream.get("repository"),
        "upstream_source_sha": upstream.get("commit"),
        "model_dir": str(model_dir),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "gpu_count": torch.cuda.device_count(),
        "package_versions": versions,
    }


def weight_map_evidence(model_dir: Path, model) -> dict[str, object]:
    """Prove the loaded module set matches the checkpoint weight map exactly."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    index_keys = set((index.get("weight_map") or {}).keys())
    model_keys = set(model.state_dict().keys())
    added = sorted(model_keys - index_keys)
    missing = sorted(index_keys - model_keys)
    return {
        "status": "PASS" if not added and not missing else "FAIL",
        "index_tensor_count": len(index_keys),
        "model_tensor_count": len(model_keys),
        "unexpected_model_tensors": added,
        "missing_model_tensors": missing,
        "weights_materialized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical MOSS-TTS v1.5 BF16 synthesis runner")
    parser.add_argument("--model-root", default="/kaggle/input/models")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    plan = generation_plan(
        args.text, args.language, seed=args.seed, max_new_tokens=args.max_new_tokens
    )

    # Identity is proven before any weights are read.
    model_dir = discover_model(Path(args.model_root))
    identity = verify_model(model_dir)
    dtype_evidence = verify_checkpoint_dtypes(model_dir, list(identity["shards"]))
    provenance = repository_provenance(model_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.synchronize()
    print("STAGE=PROCESSOR_SETUP START", flush=True)
    setup_start = time.perf_counter()
    processor = load_processor(model_dir)
    conversation = [[processor.build_user_message(text=args.text, language=args.language)]]
    batch = processor(conversation, mode="generation")
    processor_setup_seconds = time.perf_counter() - setup_start
    print(f"STAGE=PROCESSOR_SETUP DONE seconds={processor_setup_seconds:.3f}", flush=True)

    input_ids = batch["input_ids"].cpu()
    attention_mask = batch["attention_mask"].cpu()
    if getattr(processor, "audio_tokenizer", None) is not None:
        processor.audio_tokenizer = processor.audio_tokenizer.cpu()
    del processor, batch, conversation
    gc.collect()
    torch.cuda.empty_cache()
    for index in range(2):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

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
    model_load_seconds = time.perf_counter() - load_start
    print(f"STAGE=MODEL_LOAD DONE seconds={model_load_seconds:.3f}", flush=True)
    model.eval()

    tensor_map = weight_map_evidence(model_dir, model)
    if tensor_map["status"] != "PASS":
        raise RuntimeError(
            "loaded module set does not match the checkpoint weight map: "
            f"{tensor_map['unexpected_model_tensors']} / {tensor_map['missing_model_tensors']}"
        )

    input_ids = input_ids.to("cuda:0")
    attention_mask = attention_mask.to("cuda:0")
    torch.cuda.synchronize()
    print("STAGE=GENERATE START", flush=True)
    generate_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            audio_temperature=1.7,
            audio_top_p=0.8,
            audio_top_k=25,
            audio_repetition_penalty=1.0,
        )
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - generate_start
    print(f"STAGE=GENERATE DONE seconds={generate_seconds:.3f}", flush=True)
    generated_steps = count_generated_steps(outputs)

    cpu_outputs = [(int(start), ids.detach().cpu()) for start, ids in outputs]
    gpu_peaks = []
    for index in range(2):
        with torch.cuda.device(index):
            gpu_peaks.append(torch.cuda.max_memory_allocated() / 1024**3)

    del outputs, model, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    print("STAGE=DECODE START", flush=True)
    decode_start = time.perf_counter()
    processor = load_processor(model_dir)
    processor.audio_tokenizer = processor.audio_tokenizer.to("cuda:0")
    with torch.inference_mode():
        messages = processor.decode(cpu_outputs)
    decode_seconds = time.perf_counter() - decode_start
    print(f"STAGE=DECODE DONE seconds={decode_seconds:.3f}", flush=True)

    audio = messages[0].audio_codes_list[0].detach().float().cpu()
    duration = audio.numel() / float(runtime_contract()["sample_rate_hz"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio.unsqueeze(0), 24000)

    report = {
        **plan,
        "model_dir": str(model_dir),
        "model_identity": identity,
        "checkpoint_dtype_verification": dtype_evidence,
        "weight_map_match": tensor_map,
        "model_patch": False,
        "architecture_modified": False,
        "provenance": provenance,
        "sample_rate": runtime_contract()["sample_rate_hz"],
        "num_samples": int(audio.numel()),
        "wav_path": str(output_path),
        "signal": validate_waveform(audio.tolist()),
        **generation_metrics(
            processor_setup_seconds=processor_setup_seconds,
            model_load_seconds=model_load_seconds,
            generate_seconds=generate_seconds,
            decode_seconds=decode_seconds,
            audio_duration_seconds=duration,
            generated_steps=generated_steps,
            gpu_peaks_gib=gpu_peaks,
        ),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    assert report["model_identity"]["status"] == "PASS"
    assert report["checkpoint_dtype_verification"]["status"] == "PASS"
    assert report["weight_map_match"]["status"] == "PASS"
    assert report["signal"]["status"] == "PASS"
    print("MOSS_TTS_GENERATION=PASS")


if __name__ == "__main__":
    sys.exit(main())