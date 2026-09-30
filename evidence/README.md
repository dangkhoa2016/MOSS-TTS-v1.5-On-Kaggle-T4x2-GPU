# Evidence

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

`prepublication-canonical-smoke.json` records a canonical runner smoke on the development Kaggle T4x2 session. It demonstrates that the repository code path can load the attached qualified model input, generate tokens and staged-decode a valid WAV.

Its limits are explicit: the run predates the strict identity and dtype verification now in the codebase, the attached digests are not cryptographically verified, and it was produced in a warm development session rather than a fresh `Restart Session -> Run All`.

It is **not the final** clean-room publication authority. Final release evidence must come from a fresh Kaggle `Restart Session -> Run All` execution of the canonical production notebook, pinned to an immutable source revision, plus a separate human listening review. See [docs/evidence-index.md](../docs/evidence-index.md).
