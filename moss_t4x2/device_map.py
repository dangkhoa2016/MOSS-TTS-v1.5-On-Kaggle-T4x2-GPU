from __future__ import annotations


def build_device_map(num_layers: int = 36) -> dict[str, int]:
    if num_layers != 36:
        raise ValueError("qualified MOSS-TTS-v1.5 topology requires exactly 36 decoder layers")
    mapping: dict[str, int] = {
        "language_model.embed_tokens": 0,
        "language_model.rotary_emb": 0,
        "language_model.norm": 0,
        "emb_ext": 0,
        "lm_heads": 0,
    }
    for i in range(num_layers):
        mapping[f"language_model.layers.{i}"] = 0 if i <= 13 else 1
    return mapping
