# Evidence

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

`prepublication-canonical-smoke.json` ghi lại một canonical runner smoke trên Kaggle T4x2 development session. Nó cho thấy code path của repository có thể load model input đã attach đã qualification, generate tokens và staged-decode thành WAV hợp lệ.

Giới hạn của nó được nêu rõ: lượt chạy này ra trước khi có identity và dtype verification nghiêm ngặt hiện nay, digest của file đã attach không được xác minh mật mã, và nó được tạo trong development session đã ấm chứ không phải fresh `Restart Session -> Run All`.

Nó **không phải** clean-room publication authority cuối cùng. Evidence phát hành cuối phải đến từ một lượt Kaggle fresh `Restart Session -> Run All` của production notebook chuẩn, được ghim vào một source revision bất biến, cộng với một đợt human listening review riêng. Xem [docs/evidence-index.vi.md](../docs/evidence-index.vi.md).
