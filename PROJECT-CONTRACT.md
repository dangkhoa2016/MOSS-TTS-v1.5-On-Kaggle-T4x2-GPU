# MOSS-TTS v1.5 on Kaggle T4x2 — Project Contract

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PROJECT-CONTRACT.vi.md)

## Baseline
- Upstream runtime: official `OpenMOSS/MOSS-TTS` source behavior.
- Upstream revision qualified during development: `934d6826b084c46a0d033402174d5f8ac4ed2519`.
- Original model: `OpenMOSS-Team/MOSS-TTS-v1.5`.
- Kaggle mirror: `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
- Precision: BF16.
- Batch size: 1.
- Sample rate: 24 kHz.

## Target topology
- `cuda:0`: embeddings, rotary embedding, final norm, external embeddings, output heads, decoder layers 0–13.
- `cuda:1`: decoder layers 14–35.
- Original MOSS Audio Tokenizer is staged after 8B generation to avoid simultaneous VRAM residency.

## Hard rules
- No GGUF.
- No INT8/INT4/AWQ/GPTQ.
- No model-weight or architecture modification.
- No upstream source patch is required for the qualified path.
- Public production qualification requires exactly two Tesla T4 GPUs.
- Raw acronym pronunciation quality is separate from runtime qualification; IPA is an upstream-supported model feature.
