# Lịch sử phát triển

> 🌐 Language / Ngôn ngữ: [English](development-history.md) | **Tiếng Việt**

Quá trình phát triển được ghi lại theo thứ tự, mỗi bước gắn với artefact mà nó tạo ra.

1. **Danh tính model.** Checkpoint được xác định là `OpenMOSS-Team/MOSS-TTS-v1.5`, safetensors BF16 nguyên bản, upstream revision `934d6826b084c46a0d033402174d5f8ac4ed2519`, mirror `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
2. **FP16 thất bại về số học.** Diagnostics trong development thấy activation overflow và non-finite logits quanh decoder layer 6/7, vì vậy BF16 là precision duy nhất được qualification.
3. **Phân bố hai GPU.** Explicit module placement trên cả hai T4, đối chiếu với weight map của safetensors.
4. **Staged codec decode.** Backbone 8B được unload trước khi MOSS Audio Tokenizer nguyên bản decode, nên hai phần không cùng nằm trong VRAM.
5. **Canonical smoke.** [`evidence/prepublication-canonical-smoke.json`](../evidence/prepublication-canonical-smoke.json) ghi lại WAV end-to-end đầu tiên từ code path của repository. Nó ra trước khi có identity verification nghiêm ngặt nên không phải release authority.
6. **Các đường ngôn ngữ và phát âm.** Nghe chấp nhận tiếng Anh trong development, code-switch Vietnamese-English và điều khiển phát âm IPA được upstream hỗ trợ.
7. **Siết lại verification.** Identity, dtype, format, filesystem và digest chuyển sang các báo cáo machine-readable; xem [qualification-matrix.vi.md](qualification-matrix.vi.md).
8. **Release acceptance.** Một lượt fresh `Restart Session -> Run All` của [`notebooks/kaggle-production-demo.ipynb`](../notebooks/kaggle-production-demo.ipynb) đã PASS trên source authority `f43518ea4b866b42a75c7b600ab146d1b9005bd8`, sau đó reviewer duyệt rõ ràng đủ bảy audio; xem evidence index.
