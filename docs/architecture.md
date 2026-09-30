# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](architecture.vi.md)

`cuda:0` owns embeddings, layers 0–13, norm, external audio embeddings and output heads. `cuda:1` owns layers 14–35. After audio-token generation the 8B backbone is unloaded before the original MOSS Audio Tokenizer decodes the waveform.
