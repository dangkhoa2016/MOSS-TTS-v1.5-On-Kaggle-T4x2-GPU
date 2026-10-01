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
  <img alt="Release state" src="https://img.shields.io/badge/Release-Pre--publication-orange">
  <img alt="Release candidate" src="https://img.shields.io/badge/State-v1.0.0--rc-blue">
</p>

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Inference qualification for the original **OpenMOSS MOSS-TTS-v1.5** BF16 safetensors checkpoint on Kaggle using two NVIDIA Tesla T4 16 GB GPUs — **no GGUF, no quantization, no model-weight modification, and no architecture modification**.

This is an **independent engineering qualification project**, not an official OpenMOSS release. Model weights are not stored in this Git repository.

**Publication is gated.** `v1.0.0` is a release candidate that has **not been tagged**, and no GitHub release exists yet. Publication requires a fresh Kaggle `Restart Session -> Run All` run of the production notebook plus a separate human listening review.

## Why this project exists

The checkpoint contains approximately **8.49B parameters** (~15.8 GiB at two bytes/parameter). This project demonstrates a qualified path that keeps the original checkpoint and uses explicit module placement across both T4s, then stages the official audio tokenizer after autoregressive generation.

## Target topology under qualification

| Component | Device |
|---|---|
| token embeddings, rotary embedding, layers 0–13, final norm, external audio embeddings, LM/audio heads | `cuda:0` |
| decoder layers 14–35 | `cuda:1` |
| precision | BF16 |
| codec | original MOSS Audio Tokenizer, staged after backbone unload |

This is explicit module placement with Accelerate hooks, **not tensor parallelism and not a claim of pooled 32 GB VRAM**.

## Measured pre-publication smoke

The canonical repository path produced a 7.12 s WAV on Kaggle T4 x2 with generation time 12.09 s, generation RTF 1.70, and peak allocated memory of approximately 7.97 GiB / 7.94 GiB on GPU0/GPU1. This record demonstrates that the repository code path loaded the attached qualified model input and produced a valid waveform; it predates the strict identity/dtype verification now in the codebase and is **not** release authority. See [evidence/README.md](evidence/README.md).

## Kaggle production demo

Import [`notebooks/kaggle-production-demo.ipynb`](notebooks/kaggle-production-demo.ipynb) into a fresh Kaggle notebook, select **GPU T4 x2**, attach model `dangkhoa2016/openmoss-team-moss-tts-v1-5`, keep Internet ON for the source checkout, set `MOSS_TTS_SOURCE_REF` to the exact 40-character candidate commit SHA (or to a release tag once one exists), then use **Restart Session → Run All**.

The notebook pins an immutable source revision, bootstraps and verifies `requirements/kaggle.txt`, and derives its scorecard from the machine-readable preflight, inventory and generation reports. It covers English, Vietnamese, Vietnamese-English code-switching, upstream-supported IPA pronunciation control, timing/RTF, peak VRAM, inline audio playback and a derived machine-readable qualification summary.

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

## What is actually verified

| Claim | Evidence |
|---|---|
| exactly two Tesla T4 GPUs | `scripts/preflight.py` normalizes device names and fails on anything but `tesla t4` |
| checkpoint identity and BF16 | `verify_model()` requires a declared `bfloat16` dtype (a missing dtype is rejected, never inferred) plus architecture, model type and shard completeness |
| stored tensor dtype | `verify_checkpoint_dtypes()` reads safetensors headers only; no weights are materialized |
| no weight/architecture modification | the loaded module set is compared against the safetensors weight map |
| no GGUF / no quantization | derived from the verified checkpoint contents, not declared |

Machine-readable markers: `GPU_T4X2=PASS`, `MODEL_IDENTITY_AND_SAFETENSORS=PASS`, `MOSS_TTS_GENERATION=PASS`, `KAGGLE_PRODUCTION_DEMO=PASS`, `HUMAN_LISTENING_REVIEW=PENDING`. A marker is printed only after the program asserted the condition it names.

`references/checkpoint-sha256.json` carries the mechanism to compare the attached checkpoint against canonical digests, but its `expected_sha256` set is still empty, so the inventory report states `UNVERIFIED`. Cryptographic proof of unmodified upstream files therefore remains an open pre-publication task.

## Precision ruling

FP16 is not qualified for this checkpoint/runtime on T4x2: development diagnostics observed activation overflow and non-finite logits around decoder layer 6/7. BF16 remained finite and matches the checkpoint-declared dtype. BF16 here is **not quantization**.

## Licensing and attribution

Repository-authored code/documentation is MIT licensed by **Đăng Khoa <i.am@dangkhoa.dev>**. Upstream OpenMOSS source, model weights, tokenizers, and codecs retain their upstream terms. See [LICENSE-NOTES.md](LICENSE-NOTES.md), [NOTICE.txt](NOTICE.txt), and [`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/).
