import hashlib
import json
from pathlib import Path
import pytest

from moss_t4x2.inventory import inventory_model, sha256_reference_manifest


def _write_safetensors(path: Path, tensors: dict[str, str]) -> Path:
    entries = {}
    offset = 0
    for name, dtype in tensors.items():
        width = {"BF16": 2, "F16": 2, "I64": 8}[dtype]
        entries[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + width]}
        offset += width
    header = json.dumps(entries).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * offset)
    return path


def _make_model(root: Path, *, shards: tuple[str, ...] = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")) -> Path:
    directory = root / "candidate"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MossTTSDelayModel"],
                "model_type": "moss_tts_delay",
                "torch_dtype": "bfloat16",
            }
        )
    )
    (directory / "processor_config.json").write_text("{}")
    weight_map = {f"layer.{i}": shard for i, shard in enumerate(shards)}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 32}, "weight_map": weight_map})
    )
    for index, shard in enumerate(shards):
        _write_safetensors(directory / shard, {f"layer.{index}.weight": "BF16"})
    return directory


def test_inventory_reports_identity_dtype_and_filesystem_evidence(tmp_path):
    model = _make_model(tmp_path)
    report = inventory_model(model)
    assert report["status"] == "PASS"
    assert report["identity"]["model_id"] == "OpenMOSS-Team/MOSS-TTS-v1.5"
    assert report["identity"]["architecture"] == "MossTTSDelayModel"
    assert report["identity"]["model_type"] == "moss_tts_delay"
    assert report["identity"]["declared_dtype"] == "bfloat16"
    assert report["format"] == {"safetensors": True, "gguf": False, "shard_suffix": ".safetensors"}
    assert report["quantization"] == "none"
    assert report["shard_count"] == 2
    assert report["checkpoint_dtype_verification"]["status"] == "PASS"
    assert report["checkpoint_dtype_verification"]["floating_dtype"] == "bfloat16"
    assert report["checkpoint_dtype_verification"]["weights_materialized"] is False
    names = set(report["files"])
    assert {"config.json", "model.safetensors.index.json", "processor_config.json"} <= names
    assert report["total_bytes"] == sum(entry["bytes"] for entry in report["files"].values())


def test_inventory_hashes_shards_with_sha256(tmp_path):
    model = _make_model(tmp_path)
    shard = model / "model-00001-of-00002.safetensors"
    report = inventory_model(model)
    assert report["files"][shard.name]["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert report["files"][shard.name]["bytes"] == shard.stat().st_size


def test_inventory_can_skip_full_sha256_recompute(tmp_path, monkeypatch):
    model = _make_model(tmp_path)
    reference = Path(__file__).resolve().parents[1] / "references" / "checkpoint-sha256.json"

    def fail_if_hashed(path):
        raise AssertionError(f"full-file sha256 should be skipped: {path}")

    monkeypatch.setattr("moss_t4x2.inventory._sha256", fail_if_hashed)
    report = inventory_model(model, reference_path=reference, verify_sha256=False)

    assert report["checkpoint_sha256_reference"]["status"] == "SKIPPED"
    assert report["checkpoint_sha256_reference"]["gating"] is False
    assert report["checkpoint_sha256_reference"]["manifest_confirmation"] == "confirmed"
    assert all("sha256" not in entry for entry in report["files"].values())
    assert report["checkpoint_dtype_verification"]["status"] == "PASS"


def test_inventory_rejects_non_bf16_checkpoint(tmp_path):
    model = _make_model(tmp_path)
    config = json.loads((model / "config.json").read_text())
    config["torch_dtype"] = "float16"
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="declared dtype"):
        inventory_model(model)


def test_inventory_rejects_quantization_config(tmp_path):
    model = _make_model(tmp_path)
    config = json.loads((model / "config.json").read_text())
    config["quantization_config"] = {"bits": 4, "method": "gptq"}
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="quantized checkpoints are out of contract"):
        inventory_model(model)


def test_inventory_rejects_gguf_payload_next_to_checkpoint(tmp_path):
    model = _make_model(tmp_path)
    (model / "moss-tts.gguf").write_bytes(b"GGUF")
    with pytest.raises(RuntimeError, match="GGUF"):
        inventory_model(model)


def test_inventory_marks_unconfirmed_sha256_reference_as_unverified(tmp_path):
    model = _make_model(tmp_path)
    reference = tmp_path / "checkpoint-sha256.json"
    reference.write_text(
        json.dumps({"schema": "moss-t4x2/checkpoint-sha256/v1", "confirmation": "pending", "expected_sha256": {}})
    )
    report = inventory_model(model, reference_path=reference)
    assert report["checkpoint_sha256_reference"]["status"] == "UNVERIFIED"
    assert report["checkpoint_sha256_reference"]["gating"] is False
    assert report["checkpoint_sha256_reference"]["manifest_confirmation"] == "pending"


def test_inventory_passes_when_attached_files_match_expected_hashes(tmp_path):
    model = _make_model(tmp_path)
    reference = tmp_path / "checkpoint-sha256.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "moss-t4x2/checkpoint-sha256/v1",
                "confirmation": "confirmed",
                "expected_sha256": {
                    name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                    for name in ["config.json", "processor_config.json", "model.safetensors.index.json"]
                },
            }
        )
    )
    report = inventory_model(model, reference_path=reference)
    reference_report = report["checkpoint_sha256_reference"]
    assert reference_report["status"] == "PASS"
    assert reference_report["gating"] is True
    assert reference_report["compared_files"] == 3
    assert reference_report["mismatched"] == []
    assert reference_report["missing"] == []


def test_inventory_rejects_mismatching_expected_hash(tmp_path):
    model = _make_model(tmp_path)
    reference = tmp_path / "checkpoint-sha256.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "moss-t4x2/checkpoint-sha256/v1",
                "confirmation": "confirmed",
                "expected_sha256": {"config.json": "0" * 64},
            }
        )
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        inventory_model(model, reference_path=reference)


def test_inventory_rejects_missing_expected_file(tmp_path):
    model = _make_model(tmp_path)
    reference = tmp_path / "checkpoint-sha256.json"
    reference.write_text(
        json.dumps(
            {
                "schema": "moss-t4x2/checkpoint-sha256/v1",
                "confirmation": "confirmed",
                "expected_sha256": {"model-00009-of-00009.safetensors": "a" * 64},
            }
        )
    )
    with pytest.raises(RuntimeError, match="missing expected files"):
        inventory_model(model, reference_path=reference)


def test_inventory_without_reference_reports_reference_absent(tmp_path):
    model = _make_model(tmp_path)
    report = inventory_model(model, reference_path=tmp_path / "absent.json")
    assert report["checkpoint_sha256_reference"] == {
        "status": "ABSENT",
        "gating": False,
        "note": "no sha256 reference manifest supplied; originality is not cryptographically proven",
    }


def test_sha256_reference_manifest_rejects_unknown_schema(tmp_path):
    manifest = tmp_path / "checkpoint-sha256.json"
    manifest.write_text(json.dumps({"schema": "something/else", "expected_sha256": {}}))
    with pytest.raises(RuntimeError, match="schema"):
        sha256_reference_manifest(manifest)


def test_sha256_reference_manifest_rejects_malformed_digest(tmp_path):
    manifest = tmp_path / "checkpoint-sha256.json"
    manifest.write_text(
        json.dumps(
            {"schema": "moss-t4x2/checkpoint-sha256/v1", "confirmation": "confirmed", "expected_sha256": {"a": "zz"}}
        )
    )
    with pytest.raises(RuntimeError, match="sha256 digest"):
        sha256_reference_manifest(manifest)


def test_repository_manifest_is_declared_confirmed_with_canonical_hashes():
    manifest = Path(__file__).resolve().parents[1] / "references" / "checkpoint-sha256.json"
    parsed = sha256_reference_manifest(manifest)
    expected = {
        "model-00001-of-00004.safetensors": "749b82d07cebea77dd15af365e2efeb1dc0dd54537f7010795fdf42f520743a9",
        "model-00002-of-00004.safetensors": "4d891c8adb8bf9c135e8a44cd9f003f1dbf8b09278ced48d5af1fe7d989efb39",
        "model-00003-of-00004.safetensors": "111013e05174e67748443eafdd4d64c051292d0e5fce2cac23a16988b661eea1",
        "model-00004-of-00004.safetensors": "951ee88cf85996ee5c39441ace6e50f52894adad5da42ad2314b9a9db7f940b8",
        "config.json": "214fc997d98f51ab57925a5939afc6280e76044198b664221622e70d098ed06e",
        "processor_config.json": "6821d4805f10fb8b5cc743c021215097e03f1a25f7c220dfc4fb0d62b5dc97e0",
        "model.safetensors.index.json": "021ef3f74a92ee33fc076dde096d4fa9b7b45c34c9bdbfe730bf79368ac86bb2",
    }
    assert parsed["confirmation"] == "confirmed"
    assert parsed["expected_sha256"] == expected
    assert parsed["development_measured_sha256"]
