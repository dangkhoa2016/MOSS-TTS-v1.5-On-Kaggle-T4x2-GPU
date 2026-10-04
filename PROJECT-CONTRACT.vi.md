# Project Contract

> 🌐 Language / Ngôn ngữ: [English](PROJECT-CONTRACT.md) | **Tiếng Việt**

Qualified path của repository này là checkpoint safetensors BF16 nguyên bản `OpenMOSS-Team/MOSS-TTS-v1.5` trên đúng hai NVIDIA Tesla T4 16 GB của Kaggle. Không GGUF, không INT8/INT4/AWQ/GPTQ, không sửa model weights, không sửa architecture và không patch upstream source để thay đổi model semantics.

Repository dùng explicit module placement và staged codec decode. Claim release chỉ hợp lệ khi được chứng minh bằng public source revision và fresh Kaggle production notebook acceptance.
