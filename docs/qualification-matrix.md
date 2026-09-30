# Qualification matrix

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](qualification-matrix.vi.md)

Release target: original BF16 safetensors on exactly two Tesla T4 GPUs.

| Claim | Status | Marker |
|---|---|---|
| exactly two Tesla T4 GPUs | verified by preflight before any load | `GPU_T4X2=PASS` |
| original checkpoint identity and BF16 | verified by strict discovery and safetensors header check | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| no GGUF / no quantization | derived from verified checkpoint contents | same as identity |
| explicit two-GPU placement | weight map compared against loaded module set | device-map report |
| staged codec decode | backbone unloaded before tokenizer decode | generation report |
| English direct synthesis | human acceptance in development | listening review |
| Vietnamese direct clean-room acceptance | **open** | fresh notebook run |
| Vietnamese-English code-switching | **open** for publication | fresh notebook run |
| cloning paths | **open** | separate experiment |
| cold-process benchmark | **open** for publication | `COLD_PROCESS_CONTROLLED_BENCHMARK=PASS` |
| canonical SHA-256 of attached files | **open** (expected digest set is empty) | `UNVERIFIED` |
| human listening review | **pending** | `HUMAN_LISTENING_REVIEW=PENDING` |

Nothing in this repository may present an open row as satisfied. Machine markers exist so a claim carries the evidence that produced it; see [../evidence/README.md](../evidence/README.md).
