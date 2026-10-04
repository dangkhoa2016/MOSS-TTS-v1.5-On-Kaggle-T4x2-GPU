from __future__ import annotations


def generation_plan(text: str, language: str, *, seed: int = 42, max_new_tokens: int = 256, dtype: str = "bfloat16") -> dict[str, object]:
    if dtype != "bfloat16": raise ValueError("qualified T4x2 path requires BF16")
    if not text.strip(): raise ValueError("text must not be empty")
    return {
        "text": text, "language": language, "seed": int(seed), "max_new_tokens": int(max_new_tokens),
        "dtype": dtype, "audio_temperature": 1.7, "audio_top_p": 0.8, "audio_top_k": 25,
        "audio_repetition_penalty": 1.0, "attn_implementation": "sdpa",
        "model_stage": "generate_then_unload", "codec_stage": "decode_after_model_unload",
    }


def processor_load_kwargs() -> dict[str, object]:
    # Transformers 5.x forwards local_files_only into this custom processor's
    # constructor, where it is rejected. Model loading remains local-only.
    return {"trust_remote_code": True}
