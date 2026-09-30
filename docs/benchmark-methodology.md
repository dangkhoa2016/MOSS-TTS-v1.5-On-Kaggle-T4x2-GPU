# Benchmark methodology

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](benchmark-methodology.vi.md)

Report every stage separately so a number is never ambiguous:

| Metric | Meaning |
|---|---|
| `processor_setup_seconds` | processor and tokenizer preparation, before the backbone is loaded |
| `model_load_seconds` | checkpoint read, dtype verification and module placement |
| `generate_seconds` | autoregressive generation only |
| `decode_seconds` | staged audio-tokenizer decode only |
| `audio_duration_seconds` | duration of the produced waveform |
| `generated_steps` | generated token steps reported by the runtime |
| `generation_rtf` | `generate_seconds / audio_duration_seconds` |
| `decode_rtf` | `decode_seconds / audio_duration_seconds` |
| `total_seconds` | end-to-end wall clock for the sample |
| `peak_memory_gib` | `torch.cuda.max_memory_allocated` per GPU |

## Cold versus steady-state

A cold storage read of a ~15.8 GiB checkpoint dominates wall clock and is **not** steady-state generation performance. `scripts/run_controlled_benchmark.py` runs each sample in a fresh process so every sample pays the same cold cost, and it emits `COLD_PROCESS_CONTROLLED_BENCHMARK=PASS` only after the process-per-sample invariant and a non-empty sample count are asserted.

Warm repeats inside one process are useful for generation-only comparison, but they must be reported as a separate, explicitly labelled set of numbers.

## Rules

- Never average a cold load into a generation-only figure.
- Report peak memory per GPU; do not sum the two T4s into a pooled VRAM claim.
- Treat RTF as sample- and text-dependent; publish the text, not just the ratio.
- Keep exploratory tuning runs out of the release evidence set.
