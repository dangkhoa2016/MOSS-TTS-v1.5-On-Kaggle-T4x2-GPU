# Reproducibility

> 🌐 Language / Ngôn ngữ: [English](reproducibility.md) | **Tiếng Việt**

## Quy trình public sau release

1. Sau khi v1.0.0 được publish, import `notebooks/kaggle-production-demo.ipynb` từ GitHub `main`.
2. Bắt đầu Kaggle session mới với **GPU T4 x2** và attach `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
3. Giữ Internet ON.
4. Để trống `MOSS_TTS_SOURCE_REF` để dùng `v1.0.0` bất biến, hoặc override bằng SHA commit chính xác 40 ký tự khi audit/tái lập. Mutable `main` bị từ chối làm runtime authority.
5. Dùng `Restart Session -> Run All`.
6. Nghe đủ bảy audio trước khi publish notebook Kaggle công khai.

Notebook ghi resolved source commit vào report và fail-closed khi thiếu prerequisite. Public Kaggle notebook là evidence reproducibility/showcase sau release; release authority của v1.0.0 là candidate evidence set trong [evidence-index.vi.md](evidence-index.vi.md).
