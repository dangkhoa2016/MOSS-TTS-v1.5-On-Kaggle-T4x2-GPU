import json
from pathlib import Path
import pytest

from moss_t4x2.contract import runtime_contract
from moss_t4x2.model_discovery import discover_model, verify_model


def _make_model(root: Path, name: str = "candidate") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["MossTTSDelayModel"], "model_type": "moss_tts_delay", "torch_dtype": "bfloat16"}))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model-00001-of-00004.safetensors"}}))
    (d / "processor_config.json").write_text("{}")
    (d / "model-00001-of-00004.safetensors").write_bytes(b"x")
    return d


def test_runtime_contract_locks_original_bf16_path():
    c = runtime_contract()
    assert c["model_id"] == "OpenMOSS-Team/MOSS-TTS-v1.5"
    assert c["precision"] == "bfloat16"
    assert c["quantization"] == "none"
    assert c["gguf"] is False
    assert c["model_patch"] is False
    assert c["sample_rate_hz"] == 24000


def test_verify_model_accepts_expected_identity(tmp_path):
    model = _make_model(tmp_path)
    info = verify_model(model)
    assert info["architecture"] == "MossTTSDelayModel"
    assert info["model_type"] == "moss_tts_delay"


def test_discover_model_rejects_ambiguous_candidates(tmp_path):
    _make_model(tmp_path, "a")
    _make_model(tmp_path, "b")
    with pytest.raises(RuntimeError, match="ambiguous"):
        discover_model(tmp_path)


def test_verify_model_rejects_wrong_architecture(tmp_path):
    model = _make_model(tmp_path)
    cfg = json.loads((model / "config.json").read_text())
    cfg["architectures"] = ["SomethingElse"]
    (model / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="MossTTSDelayModel"):
        verify_model(model)
