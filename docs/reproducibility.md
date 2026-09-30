# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](reproducibility.vi.md)

## Public post-release flow

1. After v1.0.0 is published, import `notebooks/kaggle-production-demo.ipynb` from GitHub `main`.
2. Start a fresh Kaggle session with **GPU T4 x2** and attach `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
3. Keep Internet ON.
4. Leave `MOSS_TTS_SOURCE_REF` unset to use immutable `v1.0.0`, or override it with an exact 40-character commit SHA for audit/reproduction. Mutable `main` is rejected as runtime authority.
5. Use `Restart Session -> Run All`.
6. Review all seven audio cases before publishing the Kaggle notebook publicly.

The notebook records the resolved source commit in its reports and fails closed when a prerequisite is missing. The public Kaggle notebook is post-release reproducibility/showcase evidence; v1.0.0 release authority is the candidate evidence set indexed in [evidence-index.md](evidence-index.md).
