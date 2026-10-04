# v1.0.0 release notes

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

Initial public qualification target for running the original `OpenMOSS-Team/MOSS-TTS-v1.5` BF16 safetensors checkpoint on Kaggle Tesla T4 x2.

Highlights:
- Original BF16 checkpoint; no GGUF and no quantization.
- Explicit two-GPU module placement.
- Staged model-generation then official-codec decode lifecycle.
- Kaggle production notebook with inline audio review.
- English, Vietnamese, code-switch, and native IPA-control showcase paths.
- Machine-readable preflight, model inventory, runtime metrics, and qualification markers.

Publication remains gated on a fresh Kaggle `Restart Session -> Run All` acceptance run.
