# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](reproducibility.vi.md)

## Acceptance flow

1. Start a fresh Kaggle session and select **GPU T4 x2**.
2. Attach the documented model mirror `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
3. Keep Internet ON; the notebook checks out this repository itself.
4. Set `MOSS_TTS_SOURCE_REF` to an immutable revision: a 40-character commit SHA, or a release tag once one exists. A mutable branch such as `main` is rejected, because `main` can move between the run and the review.
5. Use `Restart Session -> Run All` and wait for every cell.
6. Accept only results produced from the pinned revision and from the attached checkpoint whose inventory report passes.

## What the notebook enforces

- The pinned SHA is recorded in every report, so a result can always be traced back to code.
- `requirements/kaggle.txt` is installed by a bootstrap cell before any repository module is imported, then its hash is recorded.
- The scorecard is derived from `results/preflight.json`, `results/inventory.json` and `results/qualification.json`; qualification booleans are not hand-written in the notebook.
- A missing prerequisite fails the notebook instead of printing a PASS marker.

## Reporting

Attach `results/*.json` to any qualification claim. If the digests in `references/checkpoint-sha256.json` are still empty, state that the attached checkpoint's bytes are unverified rather than implying otherwise. See [../evidence/README.md](../evidence/README.md).
