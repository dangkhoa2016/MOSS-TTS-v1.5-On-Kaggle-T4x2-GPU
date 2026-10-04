from moss_t4x2.device_map import build_device_map


def test_device_map_assigns_every_decoder_layer_once():
    m = build_device_map(36)
    layers = [f"language_model.layers.{i}" for i in range(36)]
    assert all(k in m for k in layers)
    assert [m[k] for k in layers[:14]] == [0] * 14
    assert [m[k] for k in layers[14:]] == [1] * 22


def test_device_map_keeps_embeddings_norm_and_heads_on_gpu0():
    m = build_device_map(36)
    for key in [
        "language_model.embed_tokens",
        "language_model.rotary_emb",
        "language_model.norm",
        "emb_ext",
        "lm_heads",
    ]:
        assert m[key] == 0
