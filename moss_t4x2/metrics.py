from __future__ import annotations

import math
from typing import TypedDict

from .contract import runtime_contract


class SignalMetrics(TypedDict):
    """Waveform-level evidence for a generated buffer."""

    status: str
    peak: float
    rms: float
    clipped: bool


class RuntimeMetrics(TypedDict, total=False):
    """Stage-separated timing and memory evidence for one synthesis run."""

    processor_setup_seconds: float
    model_load_seconds: float
    generate_seconds: float
    decode_seconds: float
    audio_duration_seconds: float
    generated_steps: int
    generation_rtf: float
    decode_rtf: float
    total_seconds: float


def validate_waveform(samples) -> SignalMetrics:
    values = [float(x) for x in samples]
    if not values:
        raise ValueError("empty waveform")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("waveform contains non-finite samples")
    peak = max(abs(x) for x in values)
    rms = math.sqrt(sum(x * x for x in values) / len(values))
    return {"status": "PASS", "peak": peak, "rms": rms, "clipped": peak >= 1.0}


def generation_metrics(
    *,
    processor_setup_seconds: float,
    model_load_seconds: float,
    generate_seconds: float,
    decode_seconds: float,
    audio_duration_seconds: float,
    generated_steps: int,
    gpu_peaks_gib: list[float],
) -> RuntimeMetrics:
    """Emit one metric per documented stage so reports match the published methodology."""
    if audio_duration_seconds <= 0:
        raise ValueError("audio duration must be positive")
    if generated_steps <= 0:
        raise ValueError("generated_steps must be positive to report generation throughput")
    required_gpus = int(runtime_contract()["required_gpu_count"])
    if len(gpu_peaks_gib) != required_gpus:
        raise ValueError(f"peak allocation evidence is required for exactly two GPUs, got {len(gpu_peaks_gib)}")
    for name, value in (
        ("processor_setup_seconds", processor_setup_seconds),
        ("model_load_seconds", model_load_seconds),
        ("generate_seconds", generate_seconds),
        ("decode_seconds", decode_seconds),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative")

    out: RuntimeMetrics = {
        "processor_setup_seconds": float(processor_setup_seconds),
        "model_load_seconds": float(model_load_seconds),
        "generate_seconds": float(generate_seconds),
        "decode_seconds": float(decode_seconds),
        "audio_duration_seconds": float(audio_duration_seconds),
        "generated_steps": int(generated_steps),
        "generation_rtf": float(generate_seconds / audio_duration_seconds),
        "decode_rtf": float(decode_seconds / audio_duration_seconds),
        "total_seconds": float(
            processor_setup_seconds + model_load_seconds + generate_seconds + decode_seconds
        ),
    }
    for index, value in enumerate(gpu_peaks_gib):
        out[f"gpu{index}_peak_alloc_gib"] = float(value)  # type: ignore[typeddict-unknown-key]
    return out