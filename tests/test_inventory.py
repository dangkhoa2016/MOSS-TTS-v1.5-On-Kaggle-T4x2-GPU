import hashlib, json
from pathlib import Path
from moss_t4x2.inventory import inventory_model


def test_inventory_reports_shards_and_sha256(tmp_path):
    (tmp_path / "config.json").write_text('{"architectures":["MossTTSDelayModel"],"model_type":"moss_tts_delay","torch_dtype":"bfloat16"}')
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"abc")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"x":shard.name}}))
    report = inventory_model(tmp_path)
    assert report["status"] == "PASS"
    assert report["files"][shard.name]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert report["files"][shard.name]["bytes"] == 3
