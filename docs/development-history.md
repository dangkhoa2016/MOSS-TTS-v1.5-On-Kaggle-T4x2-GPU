# Development history

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](development-history.vi.md)

The development path is recorded here in order, with each step tied to the artefact it produced.

1. **Model identity.** The checkpoint was identified as `OpenMOSS-Team/MOSS-TTS-v1.5`, original BF16 safetensors, upstream revision `934d6826b084c46a0d033402174d5f8ac4ed2519`, mirror `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
2. **Numerical failure of FP16.** Development diagnostics observed activation overflow and non-finite logits around decoder layer 6/7, which is why BF16 is the only qualified precision.
3. **Two-GPU placement.** Explicit module placement across both T4s, verified against the safetensors weight map.
4. **Staged codec decode.** The 8B backbone is unloaded before the original MOSS Audio Tokenizer decodes, so both never hold VRAM simultaneously.
5. **Canonical smoke.** [`evidence/prepublication-canonical-smoke.json`](../evidence/prepublication-canonical-smoke.json) records the first end-to-end WAV from the repository code path. It predates strict identity verification and is not release authority.
6. **Language and pronunciation paths.** English listening acceptance in development, Vietnamese-English code-switching, and upstream-supported IPA pronunciation control.
7. **Strengthened verification.** Identity, dtype, format, filesystem and digests moved into machine-readable reports; see [qualification-matrix.md](qualification-matrix.md).
8. **Release acceptance.** A fresh `Restart Session -> Run All` of [`notebooks/kaggle-production-demo.ipynb`](../notebooks/kaggle-production-demo.ipynb) passed on source authority `f43518ea4b866b42a75c7b600ab146d1b9005bd8`, followed by explicit approval of all seven audio cases; see the evidence index.
