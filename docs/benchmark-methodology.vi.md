# Phương pháp benchmark

> 🌐 Language / Ngôn ngữ: [English](benchmark-methodology.md) | **Tiếng Việt**

Báo cáo từng stage riêng để không có con số nào bị mơ hồ:

| Metric | Ý nghĩa |
|---|---|
| `processor_setup_seconds` | chuẩn bị processor và tokenizer, trước khi load backbone |
| `model_load_seconds` | đọc checkpoint, xác minh dtype và đặt module |
| `generate_seconds` | riêng giai đoạn autoregressive generation |
| `decode_seconds` | riêng giai đoạn decode bằng audio tokenizer được staged |
| `audio_duration_seconds` | thời lượng waveform đầu ra |
| `generated_steps` | số bước token sinh ra theo runtime |
| `generation_rtf` | `generate_seconds / audio_duration_seconds` |
| `decode_rtf` | `decode_seconds / audio_duration_seconds` |
| `total_seconds` | wall clock end-to-end của một mẫu |
| `peak_memory_gib` | `torch.cuda.max_memory_allocated` theo từng GPU |

## Cold và steady-state

Đọc cold khoảng 15.8 GiB từ storage chiếm phần lớn wall clock và **không phải** hiệu năng generation steady-state. `scripts/run_controlled_benchmark.py` chạy mỗi mẫu trong một process mới để mọi mẫu chịu cùng chi phí cold, và chỉ in `COLD_PROCESS_CONTROLLED_BENCHMARK=PASS` sau khi assert điều kiện process-mới-mỗi-mẫu và số mẫu khác rỗng.

Lặp lại ấm trong cùng một process hữu ích để so sánh riêng phần generation, nhưng phải được báo cáo thành bộ số riêng, có nhãn rõ ràng.

## Quy tắc

- Không bao giờ trộn phép đo cold load vào figure chỉ đo generation.
- Báo cáo peak memory theo từng GPU; không cộng hai T4 thành một tuyên bố pooled VRAM.
- RTF phụ thuộc mẫu và câu thoại; công bố cả văn bản, không chỉ tỉ lệ.
- Giữ các lượt tinh chỉnh thử nghiệm ngoài bộ evidence phát hành.
