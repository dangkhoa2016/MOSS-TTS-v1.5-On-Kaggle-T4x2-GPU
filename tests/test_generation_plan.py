import pytest

from moss_t4x2.generation import (
    count_generated_steps,
    generation_plan,
    processor_class_reference,
    processor_load_kwargs,
)


def test_generation_plan_locks_bf16_and_staged_codec():
    p = generation_plan("Hello", "English", seed=42, max_new_tokens=256)
    assert p["dtype"] == "bfloat16"
    assert p["model_stage"] == "generate_then_unload"
    assert p["codec_stage"] == "decode_after_model_unload"
    assert p["seed"] == 42
    assert p["audio_temperature"] == 1.7
    assert p["audio_top_p"] == 0.8
    assert p["audio_top_k"] == 25
    assert p["model_patch"] is False
    assert p["upstream_patch"] is False


def test_generation_plan_rejects_non_bf16_override():
    with pytest.raises(ValueError, match="BF16"):
        generation_plan("Hello", "English", dtype="float16")


def test_generation_plan_rejects_empty_text():
    with pytest.raises(ValueError, match="must not be empty"):
        generation_plan("   ", "English")


def test_processor_load_kwargs_excludes_transformers_loader_kwargs():
    kwargs = processor_load_kwargs()
    assert kwargs == {"trust_remote_code": True}
    for leaked in ("local_files_only", "revision", "code_revision", "token", "_from_auto"):
        assert leaked not in kwargs


def test_processor_class_reference_reads_upstream_auto_map(tmp_path):
    import json
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "processing_moss_tts.MossTTSDelayProcessor"}})
    )
    assert processor_class_reference(tmp_path) == "processing_moss_tts.MossTTSDelayProcessor"


def test_processor_class_reference_requires_auto_map(tmp_path):
    import json
    (tmp_path / "processor_config.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="AutoProcessor"):
        processor_class_reference(tmp_path)


def test_processor_class_reference_rejects_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        processor_class_reference(tmp_path)


def test_load_processor_calls_resolved_class_without_loader_kwargs(tmp_path, monkeypatch):
    import json

    from moss_t4x2 import generation

    (tmp_path / "processor_config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "processing_moss_tts.MossTTSDelayProcessor"}})
    )
    captured: dict[str, object] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            captured["model_dir"] = model_dir
            captured["kwargs"] = kwargs
            return cls()

    def fake_get_class_from_dynamic_module(class_reference, model_dir, **kwargs):
        captured["class_reference"] = class_reference
        captured["resolver_kwargs"] = kwargs
        return FakeProcessor

    processor = generation.load_processor(
        tmp_path, class_resolver=fake_get_class_from_dynamic_module
    )
    assert isinstance(processor, FakeProcessor)
    assert captured["class_reference"] == "processing_moss_tts.MossTTSDelayProcessor"
    assert captured["resolver_kwargs"] == {}
    assert captured["kwargs"] == {"trust_remote_code": True}
    assert captured["model_dir"] == str(tmp_path)


class _FakeTensor:
    def __init__(self, length: int) -> None:
        self._length = length

    def numel(self) -> int:
        return self._length


class _FakeBatch:
    shape = (1, 256)


def test_count_generated_steps_uses_sequence_shape():
    assert count_generated_steps(_FakeBatch()) == 256


def test_count_generated_steps_sums_segment_outputs():
    outputs = [(0, _FakeTensor(120)), (120, _FakeTensor(136))]
    assert count_generated_steps(outputs) == 256


def test_count_generated_steps_accepts_plain_tensors():
    assert count_generated_steps([_FakeTensor(64)]) == 64


def test_count_generated_steps_rejects_empty_output():
    assert count_generated_steps([]) == 0


def test_count_generated_steps_rejects_unknown_output_type():
    with pytest.raises(TypeError, match="cannot count"):
        count_generated_steps(object())