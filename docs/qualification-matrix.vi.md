# Ma trận qualification

> 🌐 Language / Ngôn ngữ: [English](qualification-matrix.md) | **Tiếng Việt**

Release target: safetensors BF16 nguyên bản trên đúng hai GPU Tesla T4.

| Claim | Trạng thái | Marker |
|---|---|---|
| đúng hai GPU Tesla T4 | **qualified** | `GPU_T4X2=PASS` |
| identity checkpoint nguyên bản và BF16 | **qualified** | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| không GGUF / không quantization | **qualified** | suy ra từ identity evidence |
| explicit two-GPU placement | **qualified** | device-map report |
| staged codec decode | **qualified** | generation report |
| tổng hợp tiếng Anh | **qualified** | candidate run + listening review |
| tổng hợp tiếng Việt | **qualified** | candidate run + listening review |
| code-switch Việt-Anh | **qualified** | candidate run + listening review |
| native IPA pronunciation control | **qualified** | candidate run + listening review |
| SHA-256 canonical của attached files | **qualified** | `SHA256_REFERENCE=PASS` |
| human listening review | **qualified** | `HUMAN_LISTENING_REVIEW=PASS` |
| public Kaggle reproducibility/showcase run | **bước sau release** | fresh public notebook sau v1.0.0 |

Public Kaggle run sau release không chặn việc publish v1.0.0. Xem [../evidence/README.vi.md](../evidence/README.vi.md).
