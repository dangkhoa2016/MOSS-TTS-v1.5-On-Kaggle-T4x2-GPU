# Evidence index

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](evidence-index.vi.md)

Release-authority evidence is intentionally compact. Exploratory debugging logs are not release authority.

| Claim | Authoritative artefact | Machine marker |
|---|---|---|
| hardware | `results/preflight.json` | `GPU_T4X2=PASS` |
| checkpoint identity and stored dtype | `results/inventory.json` | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| cryptographic digest of upstream files | `references/checkpoint-sha256.json` + `results/inventory.json` | currently `UNVERIFIED` (empty expected set) |
| device placement | `results/device_map.json` | placement summary |
| representative audio | `results/*.wav` from a fresh session | `MOSS_TTS_GENERATION=PASS` |
| benchmark numbers | `results/benchmark.json` | `COLD_PROCESS_CONTROLLED_BENCHMARK=PASS` |
| notebook acceptance | `results/qualification.json` | `KAGGLE_PRODUCTION_DEMO=PASS` |
| human listening review | separate reviewer record | `HUMAN_LISTENING_REVIEW=PENDING` |

A marker is only published after the emitting program asserted the condition it names, and each marker is conditional on its prerequisites: `MOSS_TTS_GENERATION=PASS` requires the identity report, and `KAGGLE_PRODUCTION_DEMO=PASS` requires the generation report. See [qualification-matrix.md](qualification-matrix.md) and [evidence/README.md](../evidence/README.md).
