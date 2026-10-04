# Ghi chú phát hành v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

Mục tiêu qualification public đầu tiên cho việc chạy checkpoint safetensors BF16 nguyên bản `OpenMOSS-Team/MOSS-TTS-v1.5` trên Kaggle Tesla T4 x2.

Điểm chính:
- Checkpoint BF16 nguyên bản; không GGUF, không quantization.
- Phân bố module tường minh trên hai GPU.
- Vòng đời staged: model generate trước, official codec decode sau.
- Kaggle production notebook có audio playback để review trực tiếp.
- Các đường test English, Vietnamese, code-switch và native IPA control.
- Preflight, inventory, runtime metrics và qualification markers machine-readable.

Việc publish vẫn chỉ được chốt sau fresh Kaggle `Restart Session -> Run All`.
