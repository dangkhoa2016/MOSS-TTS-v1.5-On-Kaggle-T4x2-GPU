# Notebooks

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

`kaggle-production-demo.ipynb` là entry point fresh-session công khai. Import từ GitHub `main`, chọn GPU T4 x2, attach Kaggle model mirror đã ghi trong tài liệu, rồi dùng `Restart Session -> Run All`.

## Source pin

Notebook mặc định `MOSS_TTS_SOURCE_REF` là release tag bất biến `v1.0.0`. Khi audit/tái lập có thể override bằng SHA commit chính xác 40 ký tự. Branch mutable như `main` bị từ chối làm runtime source authority dù chính file notebook được import từ `main`.

## Kết quả tạo ra

Notebook ghi các report machine-readable cho preflight, inventory, generation và qualification cùng các WAV đã render. Hãy giữ các output này với mọi public claim về run.
