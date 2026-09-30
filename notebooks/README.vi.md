# Notebooks

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

`kaggle-production-demo.ipynb` là entry point fresh-session dành cho reviewer. Import từ public GitHub revision, chọn GPU T4 x2, attach model mirror đã ghi tài liệu, rồi dùng `Restart Session -> Run All`.

## Ghim source

Đặt `MOSS_TTS_SOURCE_REF` trước khi chạy cell đầu tiên. Nó nhận commit SHA 40 ký tự hoặc thẻ phát hành, và từ chối nhánh mutable như `main`, vì notebook phải build đúng repository mà nó tuyên bố đã kiểm thử. Xem [README](../README.vi.md) của repository.

## Notebook ghi ra gì

Notebook ghi `results/preflight.json`, `results/inventory.json`, `results/generation.json` và `results/qualification.json`, cùng các file WAV nó render. Hãy đính kèm các file JSON đó cho mọi claim về lượt chạy đó.
