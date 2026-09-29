from __future__ import annotations

import json
from pathlib import Path

_QUALIFIED_DTYPE = "bfloat16"


def generation_plan(
    text: str,
    language: str,
    *,
    seed: int = 42,
    max_new_tokens: int = 256,
    dtype: str = _QUALIFIED_DTYPE,
) -> dict[str, object]:
    if dtype != _QUALIFIED_DTYPE:
        raise ValueError("qualified T4x2 path requires BF16")
    if not text.strip():
        raise ValueError("text must not be empty")
    return {
        "text": text,
        "language": language,
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "dtype": dtype,
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
        "attn_implementation": "sdpa",
        "model_patch": False,
        "upstream_patch": False,
        "model_stage": "generate_then_unload",
        "codec_stage": "decode_after_model_unload",
    }


def count_generated_steps(outputs) -> int:
    """Count generated audio tokens for HF/MOSS generate output shapes.

    The MOSS runner keeps the whole resident model loaded, so throughput evidence is the
    generated sequence length rather than a step-by-step callback counter.
    """
    shape = getattr(outputs, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[-1])
    if isinstance(outputs, (list, tuple)):
        steps = 0
        for item in outputs:
            ids = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
            numel = getattr(ids, "numel", None)
            steps += int(numel()) if callable(numel) else len(ids)
        return steps
    raise TypeError(f"cannot count generated steps for output type {type(outputs).__name__}")


def processor_load_kwargs() -> dict[str, object]:
    """Loader kwargs accepted by the upstream MOSS processor class.

    Transformers 5.x ``AutoProcessor.from_pretrained`` does not consume loader-only
    keywords: it copies what the cached-file resolver needs into a separate dict and then
    forwards its own ``**kwargs`` into the resolved custom processor class, additionally
    injecting ``_from_auto=True``. Anything left in ``kwargs`` therefore reaches the
    upstream processor constructor, which rejects loader metadata it does not accept
    (``local_files_only``, ``revision``, ``code_revision``, ``_from_auto``). Returning only
    ``trust_remote_code`` keeps the qualified path free of loader kwargs entirely.
    """
    return {"trust_remote_code": True}


def processor_class_reference(model_dir: str | Path) -> str:
    config_path = Path(model_dir) / "processor_config.json"
    data = json.loads(config_path.read_text())
    ref = data.get("auto_map", {}).get("AutoProcessor")
    if not isinstance(ref, str) or not ref:
        raise ValueError("processor_config.json must define auto_map.AutoProcessor")
    return ref


def load_processor(model_dir: str | Path, *, class_resolver=None):
    """Load the upstream MOSS processor without the AutoProcessor kwarg leak.

    ``AutoProcessor`` is bypassed on purpose: it injects ``_from_auto`` and re-emits the
    caller kwargs into the custom processor class, and the upstream MOSS processor
    forwards unknown keywords into ``ProcessorMixin`` where loader-only metadata such as
    ``local_files_only`` or ``revision`` is rejected. Resolving the exact class declared by
    ``processor_config.json`` keeps loading local-only while passing no loader kwargs.
    """
    if class_resolver is None:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        class_resolver = get_class_from_dynamic_module

    model_dir = Path(model_dir)
    class_ref = processor_class_reference(model_dir)
    processor_cls = class_resolver(class_ref, str(model_dir))
    return processor_cls.from_pretrained(str(model_dir), **processor_load_kwargs())