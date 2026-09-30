# Chỉ mục evidence

> 🌐 Language / Ngôn ngữ: [English](evidence-index.md) | **Tiếng Việt**

Release-authority evidence được giữ gọn. Log debug thử nghiệm không phải release authority.

| Claim | Artefact authoritative | Marker machine-readable |
|---|---|---|
| hardware | `results/preflight.json` | `GPU_T4X2=PASS` |
| danh tính checkpoint và dtype lưu trữ | `results/inventory.json` | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| digest mật mã của file upstream | `references/checkpoint-sha256.json` + `results/inventory.json` | hiện là `UNVERIFIED` (expected set rỗng) |
| vị trí module trên GPU | `results/device_map.json` | placement summary |
| audio đại diện | `results/*.wav` từ fresh session | `MOSS_TTS_GENERATION=PASS` |
| số liệu benchmark | `results/benchmark.json` | `COLD_PROCESS_CONTROLLED_BENCHMARK=PASS` |
| notebook acceptance | `results/qualification.json` | `KAGGLE_PRODUCTION_DEMO=PASS` |
| nghe thủ công của người đánh giá | hồ sơ reviewer riêng | `HUMAN_LISTENING_REVIEW=PENDING` |

Marker chỉ được công bố sau khi chương trình phát ra đã assert đúng điều kiện nó đại diện, và mỗi marker phụ thuộc điều kiện trước đó: `MOSS_TTS_GENERATION=PASS` cần báo cáo identity, `KAGGLE_PRODUCTION_DEMO=PASS` cần báo cáo generation. Xem [qualification-matrix.vi.md](qualification-matrix.vi.md) và [evidence/README.vi.md](../evidence/README.vi.md).
