import json
from pathlib import Path
import pytest

from moss_t4x2.contract import runtime_contract
from moss_t4x2.model_discovery import (
    count_model_candidates,
    discover_model,
    verify_checkpoint_dtypes,
    verify_model,
)


def _write_safetensors(path: Path, tensors: dict[str, str], *, header_only_gap: int = 0) -> Path:
    """Write a minimal but structurally valid safetensors file.

    Only the 8-byte little-endian header length plus the JSON header are parsed by the
    verifier, so the payload can stay tiny while still carrying real tensor descriptors.
    """
    entries = {}
    offset = 0
    for name, dtype in tensors.items():
        width = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}[dtype]
        entries[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + width]}
        offset += width
    header = json.dumps(entries).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * (offset + header_only_gap))
    return path


def _make_model(
    root: Path,
    name: str = "candidate",
    *,
    dtype: str | None = "bfloat16",
    architecture: str = "MossTTSDelayModel",
    model_type: str = "moss_tts_delay",
    shards: tuple[str, ...] = ("model-00001-of-00001.safetensors",),
    tensor_dtypes: dict[str, str] | None = None,
    write_shards: bool = True,
    empty_weight_map: bool = False,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    config = {"architectures": [architecture], "model_type": model_type}
    if dtype is not None:
        config["torch_dtype"] = dtype
    (directory / "config.json").write_text(json.dumps(config))
    weight_map: dict[str, str] = (
        {} if empty_weight_map else {f"layer.{i}": shard for i, shard in enumerate(shards)}
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16}, "weight_map": weight_map})
    )
    (directory / "processor_config.json").write_text("{}")
    if write_shards:
        for index, shard in enumerate(shards):
            _write_safetensors(
                directory / shard,
                tensor_dtypes or {f"layer.{index}.weight": "BF16"},
            )
    return directory


def test_runtime_contract_locks_original_bf16_path():
    c = runtime_contract()
    assert c["model_id"] == "OpenMOSS-Team/MOSS-TTS-v1.5"
    assert c["precision"] == "bfloat16"
    assert c["quantization"] == "none"
    assert c["gguf"] is False
    assert c["model_patch"] is False
    assert c["sample_rate_hz"] == 24000


def test_verify_model_accepts_expected_identity(tmp_path):
    model = _make_model(tmp_path, tensor_dtypes={"layer.0.weight": "BF16", "layer.0.idx": "I64"})
    info = verify_model(model)
    assert info["status"] == "PASS"
    assert info["architecture"] == "MossTTSDelayModel"
    assert info["model_type"] == "moss_tts_delay"
    assert info["declared_dtype"] == "bfloat16"
    assert info["model_id"] == "OpenMOSS-Team/MOSS-TTS-v1.5"
    assert info["shards"] == ["model-00001-of-00001.safetensors"]


def test_verify_model_reads_dtype_fallback_key(tmp_path):
    model = _make_model(tmp_path, dtype=None)
    (model / "config.json").write_text(
        json.dumps({"architectures": ["MossTTSDelayModel"], "model_type": "moss_tts_delay", "dtype": "bfloat16"})
    )
    assert verify_model(model)["declared_dtype"] == "bfloat16"


def test_verify_model_rejects_missing_checkpoint_dtype(tmp_path):
    model = _make_model(tmp_path, dtype=None)
    with pytest.raises(RuntimeError, match="declared dtype"):
        verify_model(model)


@pytest.mark.parametrize("dtype", ["float16", "float32", "int8", "qint8"])
def test_verify_model_rejects_non_bf16_declared_dtype(tmp_path, dtype):
    model = _make_model(tmp_path, dtype=dtype)
    with pytest.raises(RuntimeError, match="declared dtype"):
        verify_model(model)


def test_verify_model_rejects_wrong_architecture(tmp_path):
    model = _make_model(tmp_path)
    config = json.loads((model / "config.json").read_text())
    config["architectures"] = ["SomethingElse"]
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="MossTTSDelayModel"):
        verify_model(model)


def test_verify_model_rejects_wrong_model_type(tmp_path):
    model = _make_model(tmp_path, model_type="llama")
    with pytest.raises(RuntimeError, match="model_type"):
        verify_model(model)


def test_verify_model_rejects_missing_referenced_shard(tmp_path):
    model = _make_model(tmp_path, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
    (model / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(RuntimeError, match="missing safetensor shards"):
        verify_model(model)


def test_verify_model_rejects_empty_weight_map(tmp_path):
    model = _make_model(tmp_path, empty_weight_map=True)
    with pytest.raises(RuntimeError, match="weight_map"):
        verify_model(model)


def test_verify_model_rejects_missing_required_file(tmp_path):
    model = _make_model(tmp_path)
    (model / "config.json").unlink()
    with pytest.raises(RuntimeError, match="missing required model files"):
        verify_model(model)


def test_verify_checkpoint_dtypes_accepts_bf16_tensors(tmp_path):
    model = _make_model(tmp_path, tensor_dtypes={"layer.0.weight": "BF16", "layer.0.bias": "BF16", "layer.0.idx": "I64"})
    evidence = verify_checkpoint_dtypes(model, ["model-00001-of-00001.safetensors"])
    assert evidence["status"] == "PASS"
    assert evidence["floating_dtype"] == "bfloat16"
    assert evidence["verified_tensor_count"] == 3
    assert evidence["non_floating_tensor_count"] == 1
    assert evidence["non_bfloat16_floating_tensors"] == []


def test_verify_checkpoint_dtypes_rejects_float16_tensors(tmp_path):
    model = _make_model(tmp_path, tensor_dtypes={"layer.0.weight": "F16"})
    with pytest.raises(RuntimeError, match="non-BF16 floating tensors"):
        verify_checkpoint_dtypes(model, ["model-00001-of-00001.safetensors"])


def test_verify_checkpoint_dtypes_rejects_float32_tensors(tmp_path):
    model = _make_model(tmp_path, tensor_dtypes={"layer.0.weight": "F32"})
    with pytest.raises(RuntimeError, match="non-BF16 floating tensors"):
        verify_checkpoint_dtypes(model, ["model-00001-of-00001.safetensors"])


def test_verify_checkpoint_dtypes_rejects_corrupt_header(tmp_path):
    model = _make_model(tmp_path)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"not-a-safetensors-file" * 8)
    with pytest.raises(RuntimeError, match="header"):
        verify_checkpoint_dtypes(model, ["model-00001-of-00001.safetensors"])


def test_discover_model_verifies_direct_model_root(tmp_path):
    model = _make_model(tmp_path, "direct", architecture="SomethingElse")
    with pytest.raises(RuntimeError, match="MossTTSDelayModel"):
        discover_model(model)


def test_discover_model_rejects_direct_root_with_missing_dtype(tmp_path):
    model = _make_model(tmp_path, "direct", dtype=None)
    with pytest.raises(RuntimeError, match="declared dtype"):
        discover_model(model)


def test_discover_model_rejects_direct_root_with_missing_shard(tmp_path):
    model = _make_model(tmp_path, "direct", shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
    (model / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(RuntimeError, match="missing safetensor shards"):
        discover_model(model)


def test_discover_model_returns_verified_direct_root(tmp_path):
    model = _make_model(tmp_path, "direct")
    assert discover_model(model) == model


def test_discover_model_rejects_ambiguous_candidates(tmp_path):
    _make_model(tmp_path, "a")
    _make_model(tmp_path, "b")
    with pytest.raises(RuntimeError, match="ambiguous"):
        discover_model(tmp_path)


def test_discover_model_ignores_invalid_candidates(tmp_path):
    good = _make_model(tmp_path, "good")
    _make_model(tmp_path, "bad", dtype="float16")
    assert discover_model(tmp_path) == good


def test_discover_model_reports_when_no_valid_candidate(tmp_path):
    _make_model(tmp_path, "bad", model_type="llama")
    with pytest.raises(RuntimeError, match="no valid MOSS-TTS-v1.5 candidate"):
        discover_model(tmp_path)


def test_discover_model_rejects_empty_root(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="no MOSS-TTS model candidate"):
        discover_model(tmp_path / "empty")


def test_count_model_candidates_lists_verified_shaped_roots(tmp_path):
    _make_model(tmp_path, "a")
    (tmp_path / "not-a-model").mkdir()
    assert count_model_candidates(tmp_path) == ["a"]