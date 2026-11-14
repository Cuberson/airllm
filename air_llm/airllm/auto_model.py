import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx

# Architectures that need a dedicated AirLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic AirLLMBaseModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "AirLLMChatGLM",
    "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
    "QWenLMHeadModel": "AirLLMQWen",
    "BaichuanForCausalLM": "AirLLMBaichuan",
    "BaiChuanForCausalLM": "AirLLMBaichuan",
    "InternLMForCausalLM": "AirLLMInternLM",
    "KimiK3ForConditionalGeneration": "AirLLMKimiK3",
    "Qwen3_5ForConditionalGeneration": "AirLLMQwen3_5",
    "Qwen4ExpForConditionalGeneration": "AirLLMQwen4Exp",
}

# On macOS every model goes through AirLLMLlamaMlx, an MLX port of the Llama decoder block, so only
# checkpoints with that layout can run there. Mistral uses the same block and weight names.
MLX_ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM"}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_architecture(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                                token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        architectures = getattr(config, "architectures", None) or []
        return architectures[0] if architectures else ""

    @classmethod
    def class_name_for_architecture(cls, arch):
        """Map a config ``architectures[0]`` to the AirLLM class that should serve it.

        Split out of :meth:`get_module_class` so the mapping is testable without downloading a
        checkpoint: choosing a class never needs the weights, only the architecture name.
        """
        return ARCH_OVERRIDES.get(arch, "AirLLMBaseModel")

    @classmethod
    def supported_architectures(cls):
        """Architectures with a dedicated AirLLM class.

        Everything else streams through ``AirLLMBaseModel``, which discovers the decoder layout
        from the config at load time. This list is therefore *not* the set of runnable models --
        it is only the set with hand-written handling for something the generic path cannot infer.
        """
        return dict(ARCH_OVERRIDES)

    @classmethod
    def describe_support(cls):
        """Human-readable summary of how a model will be served, for diagnostics.

        Written because "does AirLLM support this model" has no single yes/no answer: it depends on
        whether the decoder can be *discovered*, not on whether a class exists for it.
        """
        dedicated = ", ".join(sorted(ARCH_OVERRIDES))
        return (
            "Any decoder-only causal LM whose blocks can be found from its config streams through "
            "the generic engine. Hand-written classes exist for: " + dedicated + ".\n"
            "Verified end-to-end (output matches transformers token for token): Llama/Mistral/Qwen2/"
            "Qwen3, Gemma, Phi, GPT-2, GPT-NeoX, GPT-J, GPT-BigCode, OPT, BLOOM, Falcon, StableLM, "
            "Cohere, OLMo and anything sharing their decoder layout."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        arch = AutoModel.get_architecture(pretrained_model_name_or_path, *inputs, **kwargs)

        cls_name = cls.class_name_for_architecture(arch)
        if cls_name == "AirLLMBaseModel" and arch not in ARCH_OVERRIDES:
            print(f"using generic AirLLM streaming model for architecture: {arch or 'unknown'}")
        return "airllm", cls_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            # Check this before AirLLMLlamaMlx downloads and splits the whole checkpoint, otherwise an
            # unsupported model only fails (or gives garbage) after that is done.
            arch = AutoModel.get_architecture(pretrained_model_name_or_path, *inputs, **kwargs)
            if arch and arch not in MLX_ARCHITECTURES:
                raise ValueError(
                    f"{arch} is not supported on macOS yet. The macOS backend (MLX) only runs "
                    f"Llama-style models: {', '.join(sorted(MLX_ARCHITECTURES))}."
                )
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
