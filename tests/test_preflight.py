import pytest
from moss_t4x2.preflight import evaluate_gpu_inventory, normalize_gpu_name


def test_accepts_exact_two_t4_devices():
    result = evaluate_gpu_inventory(["Tesla T4", "Tesla T4"])
    assert result["status"] == "PASS"
    assert result["gpu_count"] == 2
    assert result["gpu_names"] == ["Tesla T4", "Tesla T4"]
    assert result["precision"] == "bfloat16"


def test_accepts_vendor_prefixed_t4_name():
    assert evaluate_gpu_inventory(["NVIDIA Tesla T4", "Tesla T4"])["status"] == "PASS"


def test_rejects_single_gpu():
    with pytest.raises(RuntimeError, match="exactly two"):
        evaluate_gpu_inventory(["Tesla T4"])


def test_rejects_three_gpus():
    with pytest.raises(RuntimeError, match="exactly two"):
        evaluate_gpu_inventory(["Tesla T4", "Tesla T4", "Tesla T4"])


def test_rejects_non_t4_gpu():
    with pytest.raises(RuntimeError, match="Tesla T4"):
        evaluate_gpu_inventory(["Tesla T4", "NVIDIA A10"])


def test_rejects_future_name_that_only_contains_t4():
    with pytest.raises(RuntimeError, match="Tesla T4"):
        evaluate_gpu_inventory(["Tesla T4", "NVIDIA RTX T4-2000"])


def test_rejects_empty_inventory():
    with pytest.raises(RuntimeError, match="exactly two"):
        evaluate_gpu_inventory([])


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("Tesla T4", "tesla t4"),
        ("  NVIDIA   Tesla  T4  ", "tesla t4"),
        ("Tesla T4-SXM4-16GB", None),
    ],
)
def test_normalize_gpu_name(raw, normalized):
    assert normalize_gpu_name(raw) == normalized