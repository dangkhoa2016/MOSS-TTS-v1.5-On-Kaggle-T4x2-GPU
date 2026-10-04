import pytest
from moss_t4x2.generation import generation_plan


def test_generation_plan_locks_bf16_and_staged_codec():
    p = generation_plan("Hello", "English", seed=42, max_new_tokens=256)
    assert p["dtype"] == "bfloat16"
    assert p["model_stage"] == "generate_then_unload"
    assert p["codec_stage"] == "decode_after_model_unload"
    assert p["seed"] == 42
    assert p["audio_temperature"] == 1.7
    assert p["audio_top_p"] == 0.8
    assert p["audio_top_k"] == 25


def test_generation_plan_rejects_non_bf16_override():
    with pytest.raises(ValueError, match="BF16"):
        generation_plan("Hello", "English", dtype="float16")


def test_processor_load_kwargs_avoid_transformers_5_custom_processor_bug():
    from moss_t4x2.generation import processor_load_kwargs
    kwargs = processor_load_kwargs()
    assert kwargs == {"trust_remote_code": True}
    assert "local_files_only" not in kwargs
