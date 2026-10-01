# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## v1.0.0

Initial public release of the reproducible Kaggle Tesla T4 x2 deployment for the original `OpenMOSS-Team/MOSS-TTS-v1.5` BF16 checkpoint.

- Preserve the original BF16 safetensors checkpoint without GGUF, Q4/Q8 quantization, model-weight rewriting, or architecture modification.
- Add model discovery, inventory, explicit two-GPU module placement, and staged generation/decode.
- Verify checkpoint identity, declared dtype, stored safetensors tensor dtype, loaded-module coverage, and exact T4 x2 topology.
- Enforce exactly two Tesla T4 16 GB GPUs and fail preflight on unsupported device configurations.
- Add the production Kaggle notebook, machine-readable qualification markers, runtime/RTF metrics, VRAM reporting, and retained executed evidence.
- Add immutable source authority through `MOSS_TTS_SOURCE_REF`, defaulting to release tag `v1.0.0` while allowing exact 40-character SHA overrides.
- Confirm the SHA-256 checkpoint reference manifest used by the release qualification.
- Qualify Vietnamese, English, bidirectional Vietnamese–English code-switching, and upstream-supported IPA pronunciation control across seven reviewer-facing cases.
- Record explicit human listening approval and publish bilingual release evidence, documentation, governance, attribution, and license boundaries.
