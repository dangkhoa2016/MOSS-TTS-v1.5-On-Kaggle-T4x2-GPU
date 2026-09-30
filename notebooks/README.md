# Notebooks

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

`kaggle-production-demo.ipynb` is the reviewer-facing fresh-session entry point. Import it from the public GitHub revision, select GPU T4 x2, attach the documented model mirror, then use `Restart Session -> Run All`.

## Source pin

Set `MOSS_TTS_SOURCE_REF` before running the first cell. It accepts a 40-character commit SHA or a release tag, and it rejects mutable branches such as `main`, because the notebook must build the repository it claims to have tested. See the repository [README](../README.md).

## What it writes

The notebook writes `results/preflight.json`, `results/inventory.json`, `results/generation.json` and `results/qualification.json`, plus the WAV files it renders. Attach those JSON files to any claim about the run.
