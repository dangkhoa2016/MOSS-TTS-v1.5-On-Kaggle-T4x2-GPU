# MOSS-TTS v1.5 trên Kaggle T4x2 — Hợp đồng dự án

> 🌐 Language / Ngôn ngữ: [English](PROJECT-CONTRACT.md) | **Tiếng Việt**

## Baseline
- Upstream runtime: hành vi source chính thức `OpenMOSS/MOSS-TTS`.
- Upstream revision đã qualification trong quá trình phát triển: `934d6826b084c46a0d033402174d5f8ac4ed2519`.
- Model gốc: `OpenMOSS-Team/MOSS-TTS-v1.5`.
- Kaggle mirror: `dangkhoa2016/openmoss-team-moss-tts-v1-5`.
- Precision: BF16.
- Batch size: 1.
- Sample rate: 24 kHz.

## Target topology
- `cuda:0`: embeddings, rotary embedding, final norm, external embeddings, output heads, decoder layers 0–13.
- `cuda:1`: decoder layers 14–35.
- MOSS Audio Tokenizer nguyên bản được staged sau khi generation 8B để tránh cùng nằm trong VRAM.

## Hard rules
- Không GGUF.
- Không INT8/INT4/AWQ/GPTQ.
- Không sửa model weights hoặc architecture.
- Không cần patch source upstream cho qualified path.
- Public production qualification yêu cầu đúng hai GPU Tesla T4.
- Chất lượng phát âm acronym thô tách khỏi runtime qualification; IPA là tính năng được model upstream hỗ trợ.

## Qualification state
Hợp đồng này là baseline kỹ thuật, không phải tuyên bố publication. `v1.0.0` chưa được tag và chỉ có thể phát hành sau fresh Kaggle `Restart Session -> Run All` cùng một đợt human listening review riêng.
