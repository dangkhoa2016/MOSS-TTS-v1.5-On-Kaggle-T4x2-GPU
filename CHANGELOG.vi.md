# Changelog

> 🌐 Language / Ngôn ngữ: [English](CHANGELOG.md) | **Tiếng Việt**

## v1.0.0

Bản phát hành công khai đầu tiên cho deployment Kaggle Tesla T4 x2 có thể tái lập của checkpoint BF16 nguyên bản `OpenMOSS-Team/MOSS-TTS-v1.5`.

- Giữ nguyên checkpoint safetensors BF16, không GGUF, không Q4/Q8 quantization, không rewrite model weight và không thay đổi kiến trúc.
- Thêm model discovery, inventory, explicit two-GPU module placement và staged generation/decode.
- Xác minh checkpoint identity, declared dtype, dtype tensor thực tế trong safetensors, loaded-module coverage và chính xác topology T4 x2.
- Bắt buộc đúng hai GPU Tesla T4 16 GB và fail preflight với các cấu hình thiết bị không được hỗ trợ.
- Thêm production Kaggle notebook, machine-readable qualification markers, runtime/RTF metrics, VRAM reporting và retained executed evidence.
- Thêm immutable source authority qua `MOSS_TTS_SOURCE_REF`, mặc định dùng release tag `v1.0.0` và vẫn cho phép override bằng SHA chính xác 40 ký tự.
- Xác nhận SHA-256 checkpoint reference manifest được dùng trong release qualification.
- Qualification tiếng Việt, tiếng Anh, code-switch Việt–Anh hai chiều và upstream-supported IPA pronunciation control trên bảy reviewer-facing case.
- Ghi nhận explicit human listening approval và phát hành bilingual release evidence, documentation, governance, attribution và license boundaries.
