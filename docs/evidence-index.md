# Evidence index

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](evidence-index.vi.md)

Release-authority evidence is intentionally compact. Exploratory debugging logs are not release authority.

| Evidence | Authority | Marker / role |
|---|---|---|
| early canonical smoke | `evidence/prepublication-canonical-smoke.json` | historical smoke; not final authority |
| checkpoint digest authority | `references/checkpoint-sha256.json` | `SHA256_REFERENCE=PASS` |
| candidate executed notebook | `evidence/moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json` + GitHub Release asset | technical candidate evidence pinned to `f43518...` |
| checkpoint identity and stored BF16 safetensors | candidate executed notebook + inventory report | `MODEL_IDENTITY_AND_SAFETENSORS=PASS` |
| notebook acceptance | candidate executed notebook qualification output | `KAGGLE_PRODUCTION_DEMO=PASS` |
| human listening review | `evidence/v1.0.0-human-listening-review.md` | `HUMAN_LISTENING_REVIEW=PASS` |
| public Kaggle notebook | fresh import/run after v1.0.0 publication | post-release reproducibility/showcase; not a release gate |

A marker is published only after its emitting program asserted the condition it names. The post-release public Kaggle notebook is an additional reproducibility artifact, not a prerequisite for publishing v1.0.0.
