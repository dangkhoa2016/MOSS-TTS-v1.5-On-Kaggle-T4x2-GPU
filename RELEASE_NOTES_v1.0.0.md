# MOSS-TTS v1.5 on Kaggle T4x2 GPU — v1.0.0

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/RELEASE_NOTES_v1.0.0.vi.md)

## Release summary

v1.0.0 provides a reproducible Kaggle **Tesla T4 x2** deployment for the original `OpenMOSS-Team/MOSS-TTS-v1.5` checkpoint in **BF16**, without GGUF, Q4/Q8 quantization, model-weight rewriting, or architecture modification.

The qualified runtime uses explicit module placement across two independent 16 GB T4 GPUs and a staged lifecycle that unloads the autoregressive backbone before loading the original MOSS Audio Tokenizer for waveform decoding.

This is an independent engineering qualification project, not an official OpenMOSS release.

## Engineering qualification

- Preserve the original BF16 safetensors checkpoint; FP16 is intentionally not qualified after development diagnostics observed activation overflow and non-finite logits.
- Place embeddings, rotary components, decoder layers 0–13, final norm, external audio embeddings, and output heads on `cuda:0`; place decoder layers 14–35 on `cuda:1`.
- Run generation and waveform decoding as a staged backbone → unload → codec lifecycle so the deployment respects two separate T4 memory budgets rather than assuming pooled VRAM.
- Fail closed on GPU topology, model identity, BF16 dtype, safetensors completeness, loaded-module coverage, and any unexpected quantization or model patching.
- Ship a production Kaggle notebook pinned to immutable release authority, with timing/RTF metrics, VRAM reporting, inline audio review, and machine-readable PASS markers.

## Qualification results

Seven reviewer-facing cases were qualified across Vietnamese, English, bidirectional Vietnamese–English code-switching, and upstream-supported IPA pronunciation control: Vietnamese narrative, Vietnamese long-form, English narrative, English technical, Vietnamese→English code-switching, English→Vietnamese code-switching, and native IPA control.

All seven retained outputs completed the full BF16 inference and waveform-decoding path on the qualified Kaggle Tesla T4 x2 runtime and were explicitly reviewed by listening. Detailed token-level pronunciation observations remain in the human-listening evidence rather than being presented as release-level capability claims.

```text
HUMAN_LISTENING_REVIEW=PASS
```

Measured candidate-run evidence:

| Metric | Result |
|---|---:|
| Synthesized audio | 86.64 s |
| Four-stage synthesis time | 449.02 s |
| Average generation RTF | ~1.410 |
| Average decode RTF | ~0.801 |
| Generation RTF range | ~1.243–1.776 |
| Peak allocated VRAM | ~8.014 GiB GPU0 / ~7.966 GiB GPU1 |

These measurements describe the qualified Kaggle T4 x2 run, not universal performance guarantees.

## Release evidence

Release assets:

- `moss-tts-v1-5-kaggle-t4x2-candidate-evidence.ipynb` — executed qualification notebook with retained outputs;
- `moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json` — evidence metadata and checksum;
- `checkpoint-sha256.json` — confirmed checkpoint digest authority;
- `v1.0.0-human-listening-review.md` — human listening acceptance record;
- `v1.0.0-human-listening-review.vi.md` — Vietnamese listening record.

The executed evidence notebook was produced from source revision:

```text
f43518ea4b866b42a75c7b600ab146d1b9005bd8
```

That SHA is evidence provenance for the retained run. The public release itself is identified by the immutable `v1.0.0` tag.

## Run on Kaggle

Import the [canonical production notebook](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/notebooks/kaggle-production-demo.ipynb) into a fresh Kaggle session, select **GPU T4 x2**, attach the [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/openmoss-team-moss-tts-v1-5), keep Internet enabled for source checkout/dependency bootstrap, and run the notebook from top to bottom.

The notebook defaults to the immutable `v1.0.0` source tag and rejects mutable refs such as `main` as runtime authority.

## Links

- [GitHub repository](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU)
- [Canonical Kaggle production notebook](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/notebooks/kaggle-production-demo.ipynb)
- [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/openmoss-team-moss-tts-v1-5)
- [Upstream Hugging Face model](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
- [v1.0.0 release page](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/releases/tag/v1.0.0)
- [Reproducibility guide](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/docs/reproducibility.md)
- [Benchmark methodology](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/docs/benchmark-methodology.md)
- [Evidence index](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/docs/evidence-index.md)
