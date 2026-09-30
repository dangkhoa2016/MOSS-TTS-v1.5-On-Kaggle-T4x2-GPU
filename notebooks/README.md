# Notebooks

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

`kaggle-production-demo.ipynb` is the public fresh-session entry point. Import it from GitHub `main`, select GPU T4 x2, attach the documented Kaggle model mirror, then use `Restart Session -> Run All`.

## Source pin

The notebook defaults `MOSS_TTS_SOURCE_REF` to immutable release tag `v1.0.0`. For audit/reproduction you may override it with an exact 40-character commit SHA. Mutable branches such as `main` are rejected as runtime source authority even when the notebook file itself was imported from `main`.

## What it writes

The notebook writes machine-readable preflight, inventory, generation and qualification reports plus the rendered WAV files. Preserve those outputs with any public claim about a run.
