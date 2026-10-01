# MOSS-TTS v1.5 trên Kaggle T4x2 GPU — v1.0.0

> 🌐 Language / Ngôn ngữ: [English](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

## Tóm tắt release

v1.0.0 cung cấp một deployment có thể tái lập trên Kaggle **Tesla T4 x2** cho checkpoint nguyên bản `OpenMOSS-Team/MOSS-TTS-v1.5` ở **BF16**, không GGUF, không Q4/Q8 quantization, không rewrite model weight và không thay đổi kiến trúc.

Runtime đã qualification sử dụng explicit module placement trên hai GPU T4 độc lập 16 GB và lifecycle theo stage: unload autoregressive backbone trước khi load MOSS Audio Tokenizer nguyên bản để decode waveform.

Đây là dự án engineering qualification độc lập, không phải release chính thức của OpenMOSS.

## Engineering qualification

- Giữ nguyên checkpoint safetensors BF16; FP16 chủ động không được qualification sau khi diagnostic phát hiện activation overflow và non-finite logits.
- Đặt embeddings, rotary components, decoder layers 0–13, final norm, external audio embeddings và output heads trên `cuda:0`; đặt decoder layers 14–35 trên `cuda:1`.
- Chạy generation và waveform decoding theo staged lifecycle backbone → unload → codec để deployment tuân thủ hai vùng VRAM T4 tách biệt, không giả định pooled VRAM.
- Fail closed nếu sai GPU topology, model identity, BF16 dtype, safetensors completeness, loaded-module coverage hoặc phát hiện quantization/model patching ngoài dự kiến.
- Cung cấp production Kaggle notebook pin vào immutable release authority, kèm timing/RTF metrics, VRAM reporting, inline audio review và machine-readable PASS markers.

## Kết quả qualification

Bảy reviewer-facing case đã được qualification trên tiếng Việt, tiếng Anh, code-switching Việt–Anh hai chiều và upstream-supported IPA pronunciation control: Vietnamese narrative, Vietnamese long-form, English narrative, English technical, code-switch Việt→Anh, code-switch Anh→Việt và native IPA control.

Toàn bộ bảy output được lưu lại đều hoàn thành đầy đủ pipeline inference BF16 và waveform decoding trên runtime Kaggle Tesla T4 x2 đã qualification, đồng thời được nghe và đánh giá trực tiếp. Các nhận xét chi tiết ở cấp token/phát âm được giữ trong human-listening evidence thay vì được trình bày như capability claim chính của release.

```text
HUMAN_LISTENING_REVIEW=PASS
```

Kết quả đo từ candidate run:

| Metric | Kết quả |
|---|---:|
| Synthesized audio | 86.64 s |
| Four-stage synthesis time | 449.02 s |
| Average generation RTF | ~1.410 |
| Average decode RTF | ~0.801 |
| Generation RTF range | ~1.243–1.776 |
| Peak allocated VRAM | ~8.014 GiB GPU0 / ~7.966 GiB GPU1 |

Các số liệu này mô tả qualified Kaggle T4 x2 run, không phải cam kết performance phổ quát.

## Release evidence

Các release asset:

- `moss-tts-v1-5-kaggle-t4x2-candidate-evidence.ipynb` — executed qualification notebook giữ nguyên output;
- `moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json` — metadata và checksum của evidence;
- `checkpoint-sha256.json` — checkpoint digest authority đã xác nhận;
- `v1.0.0-human-listening-review.md` — human listening acceptance record;
- `v1.0.0-human-listening-review.vi.md` — bản tiếng Việt của listening record.

Executed evidence notebook được tạo từ source revision:

```text
f43518ea4b866b42a75c7b600ab146d1b9005bd8
```

SHA này là evidence provenance của retained run. Public release được định danh bởi immutable tag `v1.0.0`.

## Chạy trên Kaggle

Import [canonical production notebook](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/notebooks/kaggle-production-demo.ipynb) vào một Kaggle session mới, chọn **GPU T4 x2**, attach [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/openmoss-team-moss-tts-v1-5), bật Internet để source checkout/bootstrap dependency, rồi chạy notebook từ đầu đến cuối.

Notebook mặc định pin vào immutable source tag `v1.0.0` và từ chối mutable ref như `main` làm runtime authority.

## Liên kết

- [GitHub repository](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU)
- [Canonical Kaggle production notebook](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/notebooks/kaggle-production-demo.ipynb)
- [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/openmoss-team-moss-tts-v1-5)
- [Upstream model trên Hugging Face](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
- [Trang release v1.0.0](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/releases/tag/v1.0.0)
- [Hướng dẫn tái lập](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/docs/reproducibility.vi.md)
- [Phương pháp benchmark](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/docs/benchmark-methodology.vi.md)
- [Chỉ mục evidence](https://github.com/dangkhoa2016/MOSS-TTS-v1.5-On-Kaggle-T4x2-GPU/blob/v1.0.0/docs/evidence-index.vi.md)
