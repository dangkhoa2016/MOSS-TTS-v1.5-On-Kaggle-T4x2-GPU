# MOSS-TTS v1.5 on Kaggle T4x2 GPU

<p align="center">
  <a href="https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/actions/workflows/repository-audit.yml"><img alt="Repository Audit" src="https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/actions/workflows/repository-audit.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="THIRD_PARTY_LICENSES/OPENMOSS-APACHE-2.0"><img alt="OpenMOSS upstream license" src="https://img.shields.io/badge/OpenMOSS%20upstream-Apache--2.0-orange.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-BF16-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-Tesla%20T4%20x2-76B900?logo=nvidia&logoColor=white">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-Production%20Demo-20BEFF?logo=kaggle&logoColor=white">
  <img alt="Quantization" src="https://img.shields.io/badge/Quantization-None-0A7E07">
  <img alt="Release" src="https://img.shields.io/badge/Release-v1.0.0-blue">
</p>

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Reproducible inference for the original **OpenMOSS MOSS-TTS-v1.5** BF16 safetensors checkpoint on Kaggle using two NVIDIA Tesla T4 16 GB GPUs — **no GGUF, no quantization, no model-weight modification, and no architecture modification**.

This is an **independent engineering qualification project**, not an official OpenMOSS release. Model weights are not stored in this Git repository.

## Why this project exists

The checkpoint contains approximately **8.49B parameters** (~15.8 GiB at two bytes/parameter). This project demonstrates a qualified path that keeps the original checkpoint and uses explicit module placement across both T4s, then stages the official audio tokenizer after autoregressive generation.

## Qualified topology

| Component | Device |
|---|---|
| token embeddings, rotary embedding, layers 0–13, final norm, external audio embeddings, LM/audio heads | `cuda:0` |
| decoder layers 14–35 | `cuda:1` |
| precision | BF16 |
| codec | original MOSS Audio Tokenizer, staged after backbone unload |

This is explicit module placement with Accelerate hooks, **not tensor parallelism and not a claim of pooled 32 GB VRAM**.

## Measured pre-publication smoke

The canonical repository path produced a 7.12 s WAV on Kaggle T4 x2 with generation time 12.09 s, generation RTF 1.70, and peak allocated memory of approximately 7.97 GiB / 7.94 GiB on GPU0/GPU1. This smoke is retained as development evidence; final release authority remains gated on a fresh `Restart Session -> Run All` notebook acceptance.

## Kaggle production demo

Import [`notebooks/kaggle-production-demo.ipynb`](notebooks/kaggle-production-demo.ipynb) into a fresh Kaggle notebook, select **GPU T4 x2**, attach model `dangkhoa2016/openmoss-team-moss-tts-v1-5`, keep Internet ON for source bootstrap, then use **Restart Session → Run All**.

The notebook covers English, Vietnamese, Vietnamese-English code-switching, upstream-supported IPA pronunciation control, timing/RTF, peak VRAM, inline audio playback, and a machine-readable qualification summary.

## Documents

### Start here
- [Documentation guide](docs/README.md)
- [Engineering overview](docs/engineering-overview.md)
- [Architecture](docs/architecture.md)
- [Qualification matrix](docs/qualification-matrix.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Evidence index](docs/evidence-index.md)
- [Development history](docs/development-history.md)

### Methodology and operation
- [Benchmark methodology](docs/benchmark-methodology.md)
- [Pronunciation control](docs/pronunciation-control.md)
- [Troubleshooting](docs/troubleshooting.md)

## Precision ruling

FP16 is not qualified for this checkpoint/runtime on T4x2: development diagnostics observed activation overflow and non-finite logits around decoder layer 6/7. BF16 remained finite and matches the checkpoint-declared dtype. BF16 here is **not quantization**.

## Licensing and attribution

Repository-authored code/documentation is MIT licensed by **Đăng Khoa <i.am@dangkhoa.dev>**. Upstream OpenMOSS source, model weights, tokenizers, and codecs retain their upstream terms. See [LICENSE-NOTES.md](LICENSE-NOTES.md), [NOTICE.txt](NOTICE.txt), and [`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/).
