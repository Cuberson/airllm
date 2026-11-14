from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    # Core entry points. These have no model-specific optional dependencies, so a plain
    # `import airllm` always works as long as torch/transformers are installed.
    from .airllm_base import AirLLMBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
    # The 4bit/8bit block-wise compression helpers. These were only reachable through
    # ``airllm.utils``, which broke ``from airllm import uncompress_layer_state_dict`` (the
    # documented way to inspect a compressed shard) and the project's own compression test.
    from .utils import compress_layer_state_dict, uncompress_layer_state_dict
    # The codec engine: the lossless/lossy compression layer and the planner that picks between
    # them. Exported so callers can size a checkpoint before downloading it (``suggest_codec``,
    # ``estimate_bytes``) and so a shard can be inspected without going through a model.
    from .codecs import (LOSSLESS_CODECS, SUPPORTED_CODECS, Plan, decode_state_dict,
                         encode_state_dict, estimate_bytes, plan_codecs, read_codec_meta,
                         savings_report, suggest_codec)

    # Dedicated subclasses for a handful of custom-architecture models. Some of them pull in
    # optional extras (e.g. the Baichuan tokenizer needs `sentencepiece`). Import them defensively
    # so a missing optional dependency for one niche family never breaks the whole package; the
    # generic AirLLMBaseModel path keeps working regardless.
    import warnings as _warnings

    for _name, _module in (
        ("AirLLMLlama2", ".airllm"),
        ("AirLLMChatGLM", ".airllm_chatglm"),
        ("AirLLMQWen", ".airllm_qwen"),
        ("AirLLMQWen2", ".airllm_qwen2"),
        ("AirLLMBaichuan", ".airllm_baichuan"),
        ("AirLLMInternLM", ".airllm_internlm"),
        ("AirLLMMistral", ".airllm_mistral"),
        ("AirLLMMixtral", ".airllm_mixtral"),
        ("AirLLMKimiK3", ".airllm_kimi_k3"),
        ("AirLLMQwen3_5", ".airllm_qwen3_5"),
        ("AirLLMQwen4Exp", ".airllm_qwen4_exp"),
        ("AirLLMLoRA", ".airllm_lora"),
        ("AirLLMLoRAQwen4Exp", ".airllm_lora"),
    ):
        try:
            _mod = __import__(__name__ + _module, fromlist=[_name])
            globals()[_name] = getattr(_mod, _name)
        except Exception as _e:  # noqa: BLE001 - optional family, keep package importable
            _warnings.warn(
                f"airllm: optional model class {_name} is unavailable ({_e}). "
                f"This only affects that specific model family; the generic streaming path still works."
            )

