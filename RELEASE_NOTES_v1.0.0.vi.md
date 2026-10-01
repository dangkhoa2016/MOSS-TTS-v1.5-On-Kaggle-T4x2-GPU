# Ghi chú phát hành v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

Mục tiêu qualification public đầu tiên cho việc chạy checkpoint safetensors BF16 nguyên bản `OpenMOSS-Team/MOSS-TTS-v1.5` trên Kaggle Tesla T4 x2.

Điểm chính:
- Checkpoint BF16 nguyên bản; không GGUF, không quantization.
- Xác minh danh tính checkpoint, dtype khai báo và dtype tensor thực tế trong safetensors.
- Preflight bắt buộc đúng hai Tesla T4 trước mọi lần load model.
- Phân bố module tường minh trên hai GPU.
- Vòng đời staged: model generate trước, official codec decode sau.
- Kaggle production notebook có audio playback để review trực tiếp, được ghim vào một source revision bất biến.
- Các đường test English, Vietnamese, code-switch và native IPA control.
- Preflight, inventory, runtime metrics, benchmark theo process và qualification markers machine-readable.

Trạng thái: đây là release candidate, chưa phải release đã phát hành. Việc publish vẫn bị chặn sau fresh Kaggle `Restart Session -> Run All` và một đợt human listening review riêng.
