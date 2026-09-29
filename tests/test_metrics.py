import math
import pytest

from moss_t4x2.metrics import generation_metrics, validate_waveform


def test_generation_metrics_reports_every_documented_stage():
    metrics = generation_metrics(
        processor_setup_seconds=1.5,
        model_load_seconds=90.0,
        generate_seconds=10.0,
        decode_seconds=2.0,
        audio_duration_seconds=5.0,
        generated_steps=256,
        gpu_peaks_gib=[8.0, 7.9],
    )
    assert metrics["processor_setup_seconds"] == 1.5
    assert metrics["model_load_seconds"] == 90.0
    assert metrics["generate_seconds"] == 10.0
    assert metrics["decode_seconds"] == 2.0
    assert metrics["audio_duration_seconds"] == 5.0
    assert metrics["generated_steps"] == 256
    assert metrics["generation_rtf"] == 2.0
    assert metrics["decode_rtf"] == 0.4
    assert metrics["gpu0_peak_alloc_gib"] == 8.0
    assert metrics["gpu1_peak_alloc_gib"] == 7.9
    assert metrics["total_seconds"] == 103.5


def test_generation_metrics_matches_benchmark_methodology_keys():
    metrics = generation_metrics(
        processor_setup_seconds=0.0,
        model_load_seconds=0.0,
        generate_seconds=1.0,
        decode_seconds=1.0,
        audio_duration_seconds=1.0,
        generated_steps=1,
        gpu_peaks_gib=[1.0, 0.9],
    )
    required = {
        "processor_setup_seconds",
        "model_load_seconds",
        "generate_seconds",
        "decode_seconds",
        "audio_duration_seconds",
        "generated_steps",
        "generation_rtf",
        "decode_rtf",
        "total_seconds",
        "gpu0_peak_alloc_gib",
    }
    assert required <= set(metrics)


def test_generation_metrics_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration must be positive"):
        generation_metrics(
            processor_setup_seconds=0.0,
            model_load_seconds=0.0,
            generate_seconds=1.0,
            decode_seconds=1.0,
            audio_duration_seconds=0.0,
            generated_steps=1,
            gpu_peaks_gib=[1.0],
        )


def test_generation_metrics_rejects_missing_generated_steps():
    with pytest.raises(ValueError, match="generated_steps"):
        generation_metrics(
            processor_setup_seconds=0.0,
            model_load_seconds=0.0,
            generate_seconds=1.0,
            decode_seconds=1.0,
            audio_duration_seconds=1.0,
            generated_steps=0,
            gpu_peaks_gib=[1.0],
        )


def test_generation_metrics_requires_two_gpu_peaks():
    with pytest.raises(ValueError, match="two GPUs"):
        generation_metrics(
            processor_setup_seconds=0.0,
            model_load_seconds=0.0,
            generate_seconds=1.0,
            decode_seconds=1.0,
            audio_duration_seconds=1.0,
            generated_steps=1,
            gpu_peaks_gib=[1.0],
        )


def test_validate_waveform_reports_signal_evidence():
    signal = validate_waveform([0.0, 0.5, -0.5])
    assert signal["status"] == "PASS"
    assert signal["peak"] == 0.5
    assert signal["rms"] == pytest.approx(0.4082482904638631)
    assert signal["clipped"] is False


def test_validate_waveform_flags_clipping():
    assert validate_waveform([1.0, -1.0])["clipped"] is True


def test_validate_waveform_rejects_nonfinite_or_empty():
    with pytest.raises(ValueError):
        validate_waveform([])
    with pytest.raises(ValueError):
        validate_waveform([math.nan])
    with pytest.raises(ValueError):
        validate_waveform([math.inf])
