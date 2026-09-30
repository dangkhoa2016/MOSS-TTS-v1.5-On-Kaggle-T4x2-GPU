# Kiến trúc

> 🌐 Language / Ngôn ngữ: [English](architecture.md) | **Tiếng Việt**

`cuda:0` giữ embeddings, layers 0–13, norm, external audio embeddings và output heads. `cuda:1` giữ layers 14–35. Sau khi generate audio tokens, backbone 8B được unload trước khi MOSS Audio Tokenizer nguyên bản decode waveform.
