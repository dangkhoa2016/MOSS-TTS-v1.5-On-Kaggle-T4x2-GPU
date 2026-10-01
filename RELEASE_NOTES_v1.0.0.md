# v1.0.0 release notes

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

Initial public qualification target for running the original `OpenMOSS-Team/MOSS-TTS-v1.5` BF16 safetensors checkpoint on Kaggle Tesla T4 x2.

Highlights:
- Original BF16 checkpoint; no GGUF and no quantization.
- Verified checkpoint identity, declared dtype and stored safetensors tensor dtype.
- Exact two-Tesla-T4 preflight gate before any model load.
- Explicit two-GPU module placement.
- Staged model-generation then official-codec decode lifecycle.
- Kaggle production notebook with inline audio review, pinned to an immutable source revision.
- English, Vietnamese, code-switch, and native IPA-control showcase paths.
- Machine-readable preflight, model inventory, runtime metrics, cold-process benchmark, and qualification markers.

Status: these notes are a release candidate, not a published release. Publication remains gated on a fresh Kaggle `Restart Session -> Run All` acceptance run and a separate human listening review.
