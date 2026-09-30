# Qualification matrix

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](qualification-matrix.vi.md)

Release target: original BF16 safetensors on exactly two Tesla T4 GPUs.

| Claim | Status | Marker |
|---|---|---|
| exactly two Tesla T4 GPUs | **qualified** | `GPU_T4X2=PASS` |
| original checkpoint identity and BF16 | **qualified** | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| no GGUF / no quantization | **qualified** | derived from identity evidence |
| explicit two-GPU placement | **qualified** | device-map report |
| staged codec decode | **qualified** | generation report |
| English synthesis | **qualified** | candidate run + listening review |
| Vietnamese synthesis | **qualified** | candidate run + listening review |
| Vietnamese-English code-switching | **qualified** | candidate run + listening review |
| native IPA pronunciation control | **qualified** | candidate run + listening review |
| canonical SHA-256 of attached files | **qualified** | `SHA256_REFERENCE=PASS` |
| human listening review | **qualified** | `HUMAN_LISTENING_REVIEW=PASS` |
| public Kaggle reproducibility/showcase run | **post-release follow-up** | fresh public notebook after v1.0.0 |

The post-release public Kaggle run does not gate v1.0.0 publication. See [../evidence/README.md](../evidence/README.md).
