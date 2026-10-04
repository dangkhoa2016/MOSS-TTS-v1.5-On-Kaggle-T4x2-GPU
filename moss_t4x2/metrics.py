from __future__ import annotations
import math


def validate_waveform(samples) -> dict[str, float | str]:
    values = [float(x) for x in samples]
    if not values:
        raise ValueError("empty waveform")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("waveform contains non-finite samples")
    peak = max(abs(x) for x in values)
    rms = math.sqrt(sum(x*x for x in values) / len(values))
    return {"status": "PASS", "peak": peak, "rms": rms, "clipped": peak >= 1.0}


def generation_metrics(*, generate_seconds: float, decode_seconds: float, audio_duration_seconds: float, gpu_peaks_gib: list[float]) -> dict[str, float]:
    if audio_duration_seconds <= 0: raise ValueError("audio duration must be positive")
    out = {
        "generate_seconds": float(generate_seconds),
        "decode_seconds": float(decode_seconds),
        "audio_duration_seconds": float(audio_duration_seconds),
        "generation_rtf": float(generate_seconds / audio_duration_seconds),
        "decode_rtf": float(decode_seconds / audio_duration_seconds),
    }
    for i, value in enumerate(gpu_peaks_gib): out[f"gpu{i}_peak_alloc_gib"] = float(value)
    return out
