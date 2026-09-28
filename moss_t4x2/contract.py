from __future__ import annotations


def runtime_contract() -> dict[str, object]:
    return {
        "model_id": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "kaggle_model": "dangkhoa2016/openmoss-team-moss-tts-v1-5",
        "architecture": "MossTTSDelayModel",
        "model_type": "moss_tts_delay",
        "precision": "bfloat16",
        "quantization": "none",
        "gguf": False,
        "model_patch": False,
        "upstream_patch": False,
        "sample_rate_hz": 24000,
        "num_decoder_layers": 36,
        "batch_size": 1,
        "required_gpu_name_substring": "T4",
        "required_gpu_count": 2,
    }
