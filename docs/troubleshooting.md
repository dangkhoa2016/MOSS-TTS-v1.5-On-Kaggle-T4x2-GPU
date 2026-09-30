# Troubleshooting

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](troubleshooting.vi.md)

If preflight does not report exactly two Tesla T4 devices, stop. If model discovery is ambiguous, attach only one intended model input. Do not fall back from BF16 to FP16 on this qualified path. For Transformers 5 custom processor loading, do not pass unsupported processor constructor kwargs through `AutoProcessor`.
