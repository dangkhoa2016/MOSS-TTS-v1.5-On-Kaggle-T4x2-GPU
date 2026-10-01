# Changelog

> 🌐 Language / Ngôn ngữ: [English](CHANGELOG.md) | **Tiếng Việt**

## v1.0.0 — đang chuẩn bị

`v1.0.0` **chưa được tag**. Các mục dưới đây mô tả candidate tree hiện tại, không phải một release đã phát hành.

- Định nghĩa contract inference BF16 nguyên bản trên Kaggle T4 x2.
- Thêm model discovery, inventory, explicit device map và staged generation/decode.
- Xác minh danh tính checkpoint, dtype khai báo và dtype tensor thực tế trong safetensors thay vì giả định.
- Bắt buộc đúng hai GPU Tesla T4 và fail preflight với mọi thiết bị khác.
- Thêm production notebook, qualification machine-readable và benchmark theo process.
- Thêm immutable source pin (`MOSS_TTS_SOURCE_REF`) và scorecard suy ra từ báo cáo.
- Thêm manifest SHA-256 tham chiếu, trong đó tập digest chuẩn vẫn chờ xác nhận.
- Thêm tài liệu EN/VI, governance, attribution và license boundary.
