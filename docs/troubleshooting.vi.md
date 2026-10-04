# Xử lý sự cố

> 🌐 Language / Ngôn ngữ: [English](troubleshooting.md) | **Tiếng Việt**

Nếu preflight không thấy đúng hai Tesla T4 thì dừng. Nếu model discovery mơ hồ, chỉ attach một model input đúng mục tiêu. Không fallback BF16 sang FP16 trên qualified path này. Với custom processor trên Transformers 5, không truyền kwargs không được hỗ trợ xuyên qua `AutoProcessor`.
