# Evidence

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

`prepublication-canonical-smoke.json` ghi lại một canonical-runner smoke sớm. Nó cho thấy code path của repository có thể load model input đã attach và staged-decode WAV hợp lệ, nhưng có trước các kiểm tra identity/dtype nghiêm ngặt cuối cùng nên **không phải release authority cuối**.

Bộ evidence cho v1.0.0 bổ sung candidate notebook đã chạy được ghi nhận bởi `moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json`. Notebook 5.8 MB được giữ nguyên từng byte làm GitHub Release asset thay vì đưa vào Git history. Nó pin source `f43518ea4b866b42a75c7b600ab146d1b9005bd8`, có các technical PASS marker và đi cùng `v1.0.0-human-listening-review.vi.md` (`HUMAN_LISTENING_REVIEW=PASS`).

Notebook Kaggle chia sẻ công khai cho người dùng chủ đích là một fresh reproducibility/showcase run **sau release**. Sau v1.0.0, import `notebooks/kaggle-production-demo.ipynb` từ GitHub `main`; chính notebook mặc định dùng runtime source authority là tag bất biến `v1.0.0` và từ chối mutable `main`.
