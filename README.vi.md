# MOSS-TTS v1.5 trên Kaggle T4x2 GPU

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

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Inference qualification cho checkpoint safetensors BF16 nguyên bản **OpenMOSS MOSS-TTS-v1.5** trên Kaggle với hai NVIDIA Tesla T4 16 GB — **không GGUF, không quantization, không sửa trọng số và không sửa kiến trúc model**.

Đây là **dự án engineering qualification độc lập**, không phải release chính thức của OpenMOSS. Model weights không được lưu trong Git repository này.

**Publication vẫn còn bị chặn.** `v1.0.0` là release candidate **chưa được tag** và chưa có GitHub release. Cần chạy fresh `Restart Session -> Run All` cho production notebook và một đợt nghe thủ công riêng.

## Vì sao dự án này tồn tại

Checkpoint có khoảng **8.49B tham số** (~15.8 GiB ở hai byte/tham số). Dự án giữ checkpoint nguyên bản, phân bố module tường minh trên hai T4 rồi chỉ load official audio tokenizer sau giai đoạn autoregressive generation.

## Topology mục tiêu đang qualification

| Thành phần | Thiết bị |
|---|---|
| token embeddings, rotary embedding, layers 0–13, final norm, external audio embeddings, LM/audio heads | `cuda:0` |
| decoder layers 14–35 | `cuda:1` |
| precision | BF16 |
| codec | MOSS Audio Tokenizer nguyên bản, staged sau khi unload backbone |

Đây là explicit module placement với Accelerate hooks, **không phải tensor parallelism và không tuyên bố hai T4 hợp thành một GPU 32 GB**.

## Smoke đo được trước publication

Canonical repository path đã tạo WAV 7.12 giây trên Kaggle T4 x2, generation 12.09 giây, generation RTF 1.70, peak allocated memory khoảng 7.97 GiB / 7.94 GiB trên GPU0/GPU1. Bản ghi này chứng minh code path của repository đã load được model input đã attach và tạo WAV hợp lệ; nó ra trước khi có identity/dtype verification nghiêm ngặt nên **không phải** release authority. Xem [evidence/README.vi.md](evidence/README.vi.md).

## Kaggle production demo

Import [`notebooks/kaggle-production-demo.ipynb`](notebooks/kaggle-production-demo.ipynb) vào Kaggle notebook mới, chọn **GPU T4 x2**, attach model `dangkhoa2016/openmoss-team-moss-tts-v1-5`, bật Internet để checkout source, đặt `MOSS_TTS_SOURCE_REF` bằng SHA candidate 40 ký tự (hoặc thẻ phát hành sau này), rồi dùng **Restart Session → Run All**.

Notebook ghim một source revision bất biến, cài và kiểm tra `requirements/kaggle.txt`, rồi suy ra scorecard từ các báo cáo machine-readable của preflight, inventory và generation. Notebook bao gồm English, Vietnamese, Vietnamese-English code-switch, upstream-supported IPA pronunciation control, timing/RTF, peak VRAM, audio playback trực tiếp và qualification summary machine-readable.

## Tài liệu

### Bắt đầu tại đây
- [Hướng dẫn tài liệu](docs/README.vi.md)
- [Tổng quan engineering](docs/engineering-overview.vi.md)
- [Kiến trúc](docs/architecture.vi.md)
- [Ma trận qualification](docs/qualification-matrix.vi.md)
- [Hướng dẫn tái lập](docs/reproducibility.vi.md)
- [Chỉ mục evidence](docs/evidence-index.vi.md)
- [Lịch sử phát triển](docs/development-history.vi.md)

### Phương pháp và vận hành
- [Phương pháp benchmark](docs/benchmark-methodology.vi.md)
- [Điều khiển phát âm](docs/pronunciation-control.vi.md)
- [Troubleshooting](docs/troubleshooting.vi.md)

## Điều gì thực sự được kiểm chứng

| Claim | Bằng chứng |
|---|---|
| đúng hai GPU Tesla T4 | `scripts/preflight.py` chuẩn hoá tên thiết bị và từ chối mọi tên khác `tesla t4` |
| danh tính checkpoint và BF16 | `verify_model()` bắt buộc dtype khai báo là `bfloat16` (thiếu dtype bị từ chối, không suy diễn) cùng architecture, model type và shard completeness |
| dtype của tensor thực tế | `verify_checkpoint_dtypes()` chỉ đọc header safetensors, không materialize weight |
| không sửa weight/architecture | tập module sau khi load được so với weight map của safetensors |
| không GGUF / không quantization | suy ra từ nội dung checkpoint đã verify, không phải khai báo |

Marker machine-readable: `GPU_T4X2=PASS`, `MODEL_IDENTITY_AND_SAFETENSORS=PASS`, `MOSS_TTS_GENERATION=PASS`, `KAGGLE_PRODUCTION_DEMO=PASS`, `HUMAN_LISTENING_REVIEW=PENDING`. Marker chỉ được in sau khi chương trình đã assert đúng điều kiện mà nó đại diện.

`references/checkpoint-sha256.json` có cơ chế so digest của checkpoint đã attach với digest chuẩn, nhưng `expected_sha256` còn rỗng nên báo cáo inventory ghi `UNVERIFIED`. Chứng minh mật mã rằng file là bản upstream nguyên vẹn vẫn là việc còn mở trước publication.

## Quyết định về precision

FP16 không được qualification cho checkpoint/runtime này trên T4x2: diagnostics trong development thấy activation overflow và non-finite logits quanh decoder layer 6/7. BF16 vẫn finite và khớp dtype checkpoint khai báo. BF16 ở đây **không phải quantization**.

## Giấy phép và attribution

Code/documentation do repository này viết dùng MIT License của **Đăng Khoa <i.am@dangkhoa.dev>**. OpenMOSS source, model weights, tokenizer và codec upstream giữ nguyên điều khoản upstream. Xem [LICENSE-NOTES.vi.md](LICENSE-NOTES.vi.md), [NOTICE.txt](NOTICE.txt) và [`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/).
