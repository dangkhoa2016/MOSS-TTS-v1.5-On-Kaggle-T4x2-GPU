# Evidence

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

`prepublication-canonical-smoke.json` records an early canonical-runner smoke. It demonstrates that the repository code path could load the attached model input and staged-decode a valid WAV, but it predates the strict final identity/dtype checks and is **not the final release authority**.

The v1.0.0 release evidence set adds the executed candidate notebook recorded by `moss-tts-v1-5-kaggle-t4x2-candidate-evidence.manifest.json`. The 5.8 MB notebook itself is preserved byte-for-byte as a GitHub Release asset rather than stored in Git history. It is pinned to source `f43518ea4b866b42a75c7b600ab146d1b9005bd8`, carries the technical PASS markers, and is paired with `v1.0.0-human-listening-review.md` (`HUMAN_LISTENING_REVIEW=PASS`).

The Kaggle notebook shared publicly with users is intentionally a **post-release** fresh reproducibility/showcase run. After v1.0.0, import `notebooks/kaggle-production-demo.ipynb` from GitHub `main`; the notebook itself defaults runtime source authority to immutable tag `v1.0.0` and rejects mutable `main`.
