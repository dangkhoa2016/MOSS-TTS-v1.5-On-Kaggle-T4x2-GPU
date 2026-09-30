# Chỉ mục evidence

> 🌐 Language / Ngôn ngữ: [English](evidence-index.md) | **Tiếng Việt**

Release-authority evidence được giữ gọn. Log debug thử nghiệm không phải release authority.

| Evidence | Authority | Marker / vai trò |
|---|---|---|
| canonical smoke sớm | `evidence/prepublication-canonical-smoke.json` | smoke lịch sử; không phải authority cuối |
| authority digest checkpoint | `references/checkpoint-sha256.json` | `SHA256_REFERENCE=PASS` |
| candidate notebook đã chạy | `evidence/moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json` + GitHub Release asset | technical candidate evidence pin `f43518...` |
| identity checkpoint và safetensors BF16 lưu trữ | candidate executed notebook + inventory report | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| notebook acceptance | qualification output trong candidate executed notebook | `KAGGLE_PRODUCTION_DEMO=PASS` |
| human listening review | `evidence/v1.0.0-human-listening-review.vi.md` | `HUMAN_LISTENING_REVIEW=PASS` |
| notebook Kaggle công khai | fresh import/run sau khi publish v1.0.0 | reproducibility/showcase sau release; không phải release gate |

Marker chỉ được công bố sau khi chương trình phát ra đã assert điều kiện tương ứng. Notebook Kaggle public sau release là artifact reproducibility bổ sung, không phải điều kiện tiên quyết để publish v1.0.0.
