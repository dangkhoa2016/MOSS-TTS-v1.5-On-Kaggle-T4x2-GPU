# Tái lập kết quả

> 🌐 Language / Ngôn ngữ: [English](reproducibility.md) | **Tiếng Việt**

## Quy trình acceptance

1. Mở Kaggle session mới và chọn **GPU T4 x2**.
2. Attach model mirror đã ghi tài liệu `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
3. Giữ Internet ON; notebook tự checkout chính repository này.
4. Đặt `MOSS_TTS_SOURCE_REF` là một revision bất biến: commit SHA 40 ký tự, hoặc thẻ phát hành khi có. Nhánh mutable như `main` bị từ chối, vì `main` có thể dịch chuyển giữa lúc chạy và lúc review.
5. Dùng `Restart Session -> Run All` và đợi mọi cell chạy xong.
6. Chỉ chấp nhận kết quả tạo ra từ revision đã ghim và từ checkpoint đã attach mà báo cáo inventory của nó pass.

## Điều notebook tự kiểm

- SHA đã ghim được ghi vào mọi báo cáo, nên kết quả luôn truy ngược được về code.
- `requirements/kaggle.txt` được cài bởi cell bootstrap trước khi import bất kỳ module nào của repository, rồi hash của nó được ghi lại.
- Scorecard được suy ra từ `results/preflight.json`, `results/inventory.json` và `results/qualification.json`; boolean qualification không được viết tay trong notebook.
- Thiếu điều kiện trước đó sẽ làm notebook fail thay vì in marker PASS.

## Báo cáo

Đính kèm `results/*.json` cho mọi claim qualification. Nếu các digest trong `references/checkpoint-sha256.json` vẫn rỗng, hãy nói rằng byte của checkpoint đã attach chưa được xác minh thay vì ngụ ý ngược lại. Xem [../evidence/README.vi.md](../evidence/README.vi.md).
