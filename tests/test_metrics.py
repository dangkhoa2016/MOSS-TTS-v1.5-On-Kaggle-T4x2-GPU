import math
from moss_t4x2.metrics import generation_metrics, validate_waveform


def test_generation_metrics_computes_rtf():
    m = generation_metrics(generate_seconds=10.0, decode_seconds=2.0, audio_duration_seconds=5.0, gpu_peaks_gib=[8.0, 7.9])
    assert m["generation_rtf"] == 2.0
    assert m["decode_rtf"] == 0.4
    assert m["gpu0_peak_alloc_gib"] == 8.0


def test_validate_waveform_rejects_nonfinite_or_empty():
    assert validate_waveform([0.0, 0.5, -0.5])["status"] == "PASS"
    import pytest
    with pytest.raises(ValueError): validate_waveform([])
    with pytest.raises(ValueError): validate_waveform([math.nan])
