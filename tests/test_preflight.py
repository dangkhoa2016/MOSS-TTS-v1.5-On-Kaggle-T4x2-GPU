import pytest
from moss_t4x2.preflight import evaluate_gpu_inventory


def test_accepts_exact_two_t4_devices():
    result = evaluate_gpu_inventory(["Tesla T4", "Tesla T4"])
    assert result["status"] == "PASS"
    assert result["gpu_count"] == 2


def test_rejects_single_or_wrong_gpu():
    with pytest.raises(RuntimeError, match="exactly two"):
        evaluate_gpu_inventory(["Tesla T4"])
    with pytest.raises(RuntimeError, match="Tesla T4"):
        evaluate_gpu_inventory(["Tesla T4", "NVIDIA A10"])
