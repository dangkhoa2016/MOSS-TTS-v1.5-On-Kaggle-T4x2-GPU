# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## v1.0.0 — preparation

`v1.0.0` has **not been tagged**. Entries below describe the current candidate tree, not a published release.

- Define the original-BF16 Kaggle T4 x2 inference contract.
- Add model discovery, inventory, explicit device map, and staged generation/decode.
- Verify checkpoint identity, declared dtype and stored safetensors tensor dtype instead of assuming them.
- Enforce exactly two Tesla T4 GPUs and fail preflight on any other device.
- Add production notebook, machine-readable qualification, and cold-process benchmark tooling.
- Add an immutable source pin (`MOSS_TTS_SOURCE_REF`) and report-derived notebook scorecard.
- Add a SHA-256 reference manifest whose canonical digest set is still pending confirmation.
- Add bilingual documentation, repository governance, attribution, and license boundaries.
