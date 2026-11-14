
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationMixin
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .profiler import LayeredProfiler

from .utils import clean_memory, load_layer, layer_tensor_names, load_layer_subset, \
    uncompress_layer_state_dict, \
    find_or_create_local_splitted_path, load_merged_ngram_embedding, \
    open_ngram_mmap_table, MmapEmbedding, _force_meta_embeddings, default_device, \
    find_decoder_layout, derive_layer_names_dict, has_layout_shards
from .persist import ModelPersister
from .layer_cache import StreamedLayerCache, evict_module
from .progress import ProgressReporter, Throughput

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False


# Helpers that transformers 5.0 moved out of transformers.utils.generic. Remote model code is
# routinely written against an older transformers and still imports them from the old location,
# which makes such models fail to import at all. Re-exporting is enough to load them.
_RELOCATED_TRANSFORMERS_SYMBOLS = {
    'OutputRecorder': 'transformers.utils.output_capturing',
    'check_model_inputs': 'transformers.utils.output_capturing',
}


def _format_bytes(n):
    """Human-readable size that stays truthful below a gigabyte (``0.0GB`` hides a real budget)."""
    n = float(n)
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return f"{n:.0f}B"


def restore_relocated_transformers_symbols():
    """Re-export moved transformers helpers under their old names, where they are missing."""
    import importlib
    import transformers.utils.generic as generic

    for name, new_home in _RELOCATED_TRANSFORMERS_SYMBOLS.items():
        if hasattr(generic, name):
            continue
        try:
            module = importlib.import_module(new_home)
        except ImportError:
            continue
        symbol = getattr(module, name, None)
        if symbol is not None:
            setattr(generic, name, symbol)


try:
    from transformers.generation.streamers import BaseStreamer
except Exception:  # noqa: BLE001 - very old/new transformers; progress then stays layer-based
    BaseStreamer = object


class _ProgressStreamer(BaseStreamer):
    """Counts generated tokens so the reporter can print a real tokens/second figure.

    Only installed when ``progress=True``; it never modifies the token stream, it only observes.
    ``generate(streamer=...)`` would make transformers return a generator instead of a result, so
    this is attached after generation starts via a callback instead of being passed in. When that
    is not possible the reporter still ticks on every layer load, which is the case that looks
    hung in practice.
    """

    def __init__(self, reporter, throughput):
        self.reporter = reporter
        self.throughput = throughput
        self.tokens = 0

    def put(self, value):
        try:
            n = int(value.shape[-1]) if hasattr(value, 'shape') else 1
        except Exception:  # noqa: BLE001
            n = 1
        self.tokens += n
        self.throughput.tick(n)

    def end(self):
        pass


class AirLLMBaseModel:
    """
    Memory-frugal wrapper around a Hugging Face ``*ForCausalLM`` model.

    The checkpoint is split into per-layer shards on disk. The real transformers model is
    instantiated on the ``meta`` device (no memory used) and owns the full forward / generation
    logic. AirLLM only attaches forward hooks to each big module (embeddings, every decoder
    layer, the final norm and the lm_head) to stream that module's weights disk -> GPU right
    before it runs and free them right after, prefetching the next module on a worker thread.

    Because transformers drives the forward pass, AirLLM no longer needs to track per-architecture
    attention/rotary/cache details: new model architectures work as soon as transformers supports
    them.
    """

    # Upper bound on how much pinned (page-locked) host memory a single prefetched layer may use.
    # Layers larger than this are loaded into ordinary pageable memory instead.
    max_pinned_layer_bytes = 2 * 1024 ** 3

    # Subclasses override this to point at non-standard module names.
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                                 'layer_prefix': 'model.layers',
                                 'norm': 'model.norm',
                                 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device=None, dtype=None, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 install_hooks=True, load_resident=True,
                 layer_cache_gb=None, park_cache_gb=None, prefetch_depth=None,
                 codec=None, size_budget_gb=None, allow_lossy=None,
                 progress=False, progress_interval=5.0, stall_timeout=None,
                 cache_plan_path=None, full_load=None):
        """
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            runtime device. By default the best available is picked: CUDA, else Apple MPS,
            else CPU. Pass an explicit value (e.g. "cuda:1", "cpu") to override.
        dtype : torch.dtype, optional
            runtime dtype; defaults to the model's own config.torch_dtype (usually bfloat16 for
            modern models). float16 has too narrow a range for very deep models and overflows to
            inf/NaN, which silently corrupts the output, so we don't force it.
        max_seq_len : int, optional
            max seq length, by default 512
        layer_shards_saving_path : str, optional
            optional path to save the splitted shards, by default next to the model cache
        profiling_mode : bool, optional
            whether to profile the model loading time, default False
        compression: str, optional
            '4bit' or '8bit' to enable block-wise quantization of the on-disk shards
        codec : str, optional
            Codec used when writing the layer shards. ``'none'`` (default) stores them as they
            arrive. ``'lossless'``/``'exponent'``/``'nvcomp'`` entropy-code the float exponents and
            reconstruct the checkpoint bit for bit, so the model's output is provably unchanged
            (~1.5x smaller on bf16). ``'fp8'``/``'4bit'``/``'8bit'``/``'ternary'`` re-encode the
            weights and therefore change the output; they are only selected when asked for by name.
            ``'auto'`` picks the strongest lossless codec, and will not go lossy unless
            ``size_budget_gb`` is set and ``allow_lossy`` is not False. Each codec writes to its own
            ``splitted_model.codec-<name>`` directory, so switching never reuses a stale split.
        size_budget_gb : float, optional
            Target size for the shards. With ``codec='auto'`` this is what lets the planner trade
            fidelity for space; it walks lossless -> fp8 -> 4bit -> ternary and stops at the first
            codec that fits. Without it, planning stays lossless.
        allow_lossy : bool, optional
            Set false to forbid any output-changing codec even when a budget cannot otherwise be
            met (the planner then returns the smallest lossless plan and reports the shortfall).
        hf_token: str, optional
            huggingface api token
        prefetching: bool, optional
            overlap the next layer's disk load with the current layer's compute
        delete_original: bool, optional
            delete the original downloaded checkpoint after splitting to save disk space
        install_hooks: bool, optional
            attach inference streaming hooks. Training drives load/evict itself, so LoRA
            training passes False — the hooks call ``module.to('meta')`` after forward and
            that breaks backward.
        load_resident: bool, optional
            load ``resident`` modules (e.g. a vision tower) onto the GPU. Text-only LoRA
            training leaves them on meta.
        layer_cache_gb : float, optional
            VRAM budget, in GB, for keeping recently used layers materialised on the device.
            Default: auto — a conservative fraction of the free memory on the running device
            (0 on CPU, where streaming already goes straight to pageable host memory). Cached
            layers make repeat passes free, which is what turns generation from "re-read the
            whole checkpoint per token" into "read it once". Pass 0 to disable.
        park_cache_gb : float, optional
            Pinned-host budget, in GB, for keeping an evicted layer's host copy so it can be
            restored with a memcpy instead of a disk read. Costs no VRAM, so this is the tier
            that helps on cards too small for the VRAM cache. Default: auto (a fraction of RAM,
            capped); pass 0 to disable.
        prefetch_depth : int, optional
            How many layers to read ahead. Default 2. One is not enough to overlap anything: the
            reader is only handed the next layer once the current one has been consumed, so the
            main thread waits for that read instead of computing through it. Two gives the reader
            a full layer of lookahead. Higher values help when a read is much slower than a
            layer's compute (deep prefetch costs host memory: one layer per queued read). Ignored
            when prefetching is off.
        progress : bool, optional
            print stage transitions and periodic heartbeats (to stderr) so a long run is clearly
            distinguishable from a hang. Off by default; cheap when off.
        progress_interval : float, optional
            seconds between heartbeats when ``progress`` is on, default 5.
        stall_timeout : float, optional
            raise an error if a single stage makes no progress for this many seconds. Off by
            default. Useful on network filesystems, where a dead mount otherwise hangs forever.
        full_load : bool, optional
            load the whole model onto the device instead of streaming it. Default: auto — if
            every module fits in the memory available to this process at start-up, the model is
            loaded once and generation runs at normal transformers speed. A model that does not
            fit streams exactly as before, so this can only ever make things faster. Pass False
            to force streaming (e.g. to measure it, or to leave VRAM for another process).
        cache_plan_path : str, optional
            JSON file recording which modules were made resident in a previous run. When the
            same device and checkpoint are seen again, that plan is replayed instead of
            re-deriving the budget, so a warm start spends its VRAM exactly as the last run left
            it. Written on exit of the first run; safe to delete at any time.
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self.hf_quantizer = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                              'To use compression, please install bitsandbytes: `pip install bitsandbytes`')

        self.compression = compression
        self.hf_token = hf_token

        # Caching configuration. The budgets are resolved later, once the running device is known
        # and the per-layer byte size can be measured (a shard file size, or the meta parameters).
        self.layer_cache_gb = layer_cache_gb
        self.park_cache_gb = park_cache_gb
        self.prefetch_depth = prefetch_depth
        self.layer_cache = StreamedLayerCache()
        self.cache_plan_path = cache_plan_path
        # Populated by _replay_cache_plan when a plan matches this run; empty otherwise.
        self._planned_resident = set()
        self._planned_order = []
        self._plan_primed = True
        self.full_load = full_load
        self.full_loaded = False
        self.reporter = ProgressReporter(enabled=progress, interval=progress_interval)
        self.stall_timeout = stall_timeout
        # Counts modules streamed in the current pass, so heartbeats can report a real rate.
        self.throughput = Throughput()
        if compression is not None and layer_cache_gb not in (None, 0):
            # Caching a compressed layer still works -- the cached copy is the expanded module --
            # but a budget the user sized by file size will hold fewer layers than expected, so
            # say it rather than letting the cache look ineffective.
            print(f"compression={compression}: cached layers are held expanded, so a budget of "
                  f"{layer_cache_gb}GB holds roughly "
                  f"{layer_cache_gb / (2 if compression == '8bit' else 4):.2f}GB of checkpoint.")
        restore_relocated_transformers_symbols()

        self.set_layer_names_dict()

        # Read the config first: the splitter needs the architecture to lay out a checkpoint whose
        # trunk is not called ``model`` (GPT-NeoX / BLOOM / Falcon / GPT-2 / OPT). Loading it here
        # also lets the runtime reuse it instead of parsing the same JSON twice.
        config = self._read_config(model_local_path_or_repo_id, hf_token)
        if config is not None:
            self.config = config
            self.trust_remote_code = self._config_used_remote_code

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original,
            config=config,
            compression_device=device if device is not None else default_device(),
            codec=codec, size_budget_gb=size_budget_gb, allow_lossy=allow_lossy)

        self.running_device = device if device is not None else default_device()
        self.device = torch.device(self.running_device)

        if getattr(self, 'config', None) is None:
            config = self._read_config(self.model_local_path, hf_token)
            if config is None:
                raise RuntimeError(f"could not read a model config from {self.model_local_path}")
            self.config = config
            self.trust_remote_code = self._config_used_remote_code

        # A split directory made before a layout change (or by an older AirLLM) can carry the
        # default shard names for a model that needs a different layout. Re-derive rather than
        # failing on a missing file, so upgrading does not require deleting a 1TB split by hand.
        if not has_layout_shards(self.layer_names_dict, self.checkpoint_path):
            derived = derive_layer_names_dict(self.layer_names_dict, self.config,
                                              self.checkpoint_path)
            if derived is not self.layer_names_dict and derived:
                self.layer_names_dict = derived

        # Default to the model's native dtype (bf16 for most modern models). Forcing fp16 overflows
        # on deep models (e.g. Qwen3-235B's 94 layers) and produces garbage; bf16's wider range
        # avoids it. Users can still override via dtype=.
        if dtype is None:
            cfg_dtype = getattr(self.config, "torch_dtype", None)
            if cfg_dtype is None:
                cfg_dtype = getattr(self.config, "dtype", None)
            if cfg_dtype is None:
                text_cfg = getattr(self.config, "text_config", None)
                cfg_dtype = getattr(text_cfg, "torch_dtype", None) or getattr(text_cfg, "dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.float16
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        # prefetch executor / state. Compression used to force this off, because a prefetch that
        # also decompressed would keep two full-size layers alive at once -- exactly what
        # compression exists to avoid. The prefetch thread therefore reads the *packed* shard and
        # decompresses on the consumer side (see _load_streamed_layer / _take_prefetch).
        self.prefetching = prefetching
        self._executor = ThreadPoolExecutor(max_workers=1) if self.prefetching else None
        self._prefetch_future = None
        self._prefetched_idx = None

        self.init_model()
        # compute layer count from the instantiated model
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)

        self.layer_names = ([self.layer_names_dict['embed']]
                            + list(self.layer_names_dict.get('prefix_modules', []))
                            + [f'{self.layer_names_dict["layer_prefix"]}.{i}'
                               for i in range(layers_count)]
                            + [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']])

        self.max_seq_len = max_seq_len

        self.set_layers_from_layer_names()
        if load_resident:
            self._load_resident_modules()
        self._load_cpu_resident_modules()
        if install_hooks:
            self._install_streaming_hooks()

    def _resolve_layout(self, model):
        """Derive the streaming layout from the instantiated model's tree.

        The default layout (``model.embed_tokens`` / ``model.layers`` / ``model.norm``) covers the
        Llama/Qwen/Mistral/Gemma family. Plenty of popular architectures keep the same decoder
        behind different attribute names -- GPT-NeoX uses ``gpt_neox.layers`` + ``gpt_neox.embed_in``,
        BLOOM/Falcon/GPT-2 use ``transformer.h``, Falcon/GPT-2/Mpt have no final norm at the trunk
        level. Requiring a hand-written subclass for each of those is what kept users from running
        them at all, so the names are discovered from the module tree instead.

        Discovery is conservative: it only *fills in* what the subclass left at the default, and
        only when a complete, streamable layout is found. Anything unexpected keeps the original
        names, and the construct-time error then names the missing module.

        Returns True when the names were changed.
        """
        current = self.layer_names_dict
        try:
            model_attr = model
            for attr in current['layer_prefix'].split('.'):
                model_attr = getattr(model_attr, attr)
            if len(model_attr) > 0:
                return False  # the declared layout already works
        except (AttributeError, TypeError):
            pass

        decoder = find_decoder_layout(model)
        if decoder is None:
            return False

        found = dict(current)
        found['layer_prefix'] = decoder['layers']
        if decoder.get('embed'):
            found['embed'] = decoder['embed']
        if decoder.get('norm'):
            found['norm'] = decoder['norm']
        else:
            # No final norm (GPT-2, Falcon): point at a zero-parameter module so the streaming
            # sequence still has a slot, rather than omitting a stage the hooks rely on.
            found['norm'] = decoder['fallback_norm']
            found['stateless_norm'] = True
        found.setdefault('lm_head', 'lm_head')
        prefix_modules = decoder.get('prefix_modules') or []
        if prefix_modules:
            found['prefix_modules'] = prefix_modules

        changed = {k: v for k, v in found.items() if current.get(k) != v}
        if changed:
            print(f"derived streaming layout for {type(model).__name__}: "
                  + ", ".join(f"{k}={v!r}" for k, v in changed.items()))
        self.layer_names_dict = found
        return True

    # ---- customization hooks for subclasses -------------------------------------------------

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def _read_config(self, path, hf_token):
        """Load the model config, preferring transformers' native implementation.

        Only trust the model's bundled remote code when transformers doesn't recognize the
        architecture: vendored remote code is frequently pinned to an old transformers and breaks
        against the current cache/generation APIs (DeepSeek-V2's modeling_deepseek.py, for
        instance, calls the long-removed ``DynamicCache.seen_tokens``). Returns None when no config
        can be read at all.
        """
        token_kwargs = {'token': hf_token} if hf_token is not None else {}
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=False, **token_kwargs)
            self._config_used_remote_code = False
            return config
        except Exception:  # noqa: BLE001 - fall through to remote code
            pass
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=True, **token_kwargs)
            self._config_used_remote_code = True
            return config
        except Exception:  # noqa: BLE001 - not a loadable model (yet)
            return None

    # ---- model construction -----------------------------------------------------------------

    def _propagate_attn_implementation(self, impl):
        """Push the attention choice down into nested sub-configs.

        Multimodal wrappers keep the real decoder under a sub-config -- Kimi K3 uses ``text_config``
        -- and transformers records the request only on the config it was handed. The sub-model then
        reads an unset value and falls through to a flash-attention path, which fails outright on a
        machine without flash-attn installed.
        """
        from transformers import PretrainedConfig

        def walk(cfg, depth=0):
            if depth > 2:
                return
            for sub in vars(cfg).values():
                if isinstance(sub, PretrainedConfig):
                    sub._attn_implementation = impl
                    walk(sub, depth + 1)

        walk(self.config)

    def _model_matches_layer_names(self, model):
        """True when the instantiated class actually has the modules we plan to stream.

        Auto factories key off ``model_type``, so a VL checkpoint can land on a text-only or
        backbone class whose tree doesn't match ``layer_names_dict``. Reject those and keep trying.
        """
        try:
            for key in ('embed', 'layer_prefix', 'norm', 'lm_head'):
                mod = model
                for attr in self.layer_names_dict[key].split('.'):
                    mod = getattr(mod, attr)
            return True
        except AttributeError:
            return False

    def _auto_model_classes(self):
        """Auto* factories to try, in order, when building the empty model.

        ``AutoModelForCausalLM`` maps some VL model_types (notably ``qwen3_5``) onto a text-only
        ``*ForCausalLM`` class. That class expects a text config and either crashes or builds a
        model whose module names don't match the checkpoint. Conditional-generation architectures
        therefore try the image-text factories first.
        """
        archs = getattr(self.config, "architectures", None) or []
        arch = archs[0] if archs else ""
        names = []
        if any(tag in arch for tag in ("ConditionalGeneration", "ImageTextToText", "Multimodal")):
            names.extend(["AutoModelForImageTextToText", "AutoModelForMultimodalLM"])
        names.extend([
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
            "AutoModel",
        ])
        seen = set()
        classes = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            cls = getattr(transformers, name, None)
            if cls is not None:
                classes.append((name, cls))
        return classes

    def _instantiate_on_meta(self, attn_implementation):
        """Build the transformers model on the meta device for one attention implementation."""
        self._propagate_attn_implementation(attn_implementation)
        kwargs = {
            "attn_implementation": attn_implementation,
            "trust_remote_code": self.trust_remote_code,
        }
        errors = []
        with _force_meta_embeddings():
            for name, cls in self._auto_model_classes():
                try:
                    with init_empty_weights(include_buffers=False):
                        model = cls.from_config(self.config, **kwargs)
                except TypeError:
                    # Older Auto factories don't take attn_implementation.
                    try:
                        with init_empty_weights(include_buffers=False):
                            model = cls.from_config(self.config, trust_remote_code=self.trust_remote_code)
                    except Exception as e:  # noqa: BLE001 - try the next factory
                        errors.append(f"{name}: {type(e).__name__}: {e}")
                        continue
                except Exception as e:  # noqa: BLE001 - try the next factory
                    errors.append(f"{name}: {type(e).__name__}: {e}")
                    continue
                if not self._model_matches_layer_names(model):
                    # The declared layout may simply be the default one for a family that names
                    # its trunk differently (GPT-NeoX, BLOOM, Falcon, GPT-2, OPT). Try deriving
                    # the real layout from this tree once before giving up on the factory.
                    if self._resolve_layout(model) and self._model_matches_layer_names(model):
                        print(f"built empty {type(model).__name__} via {name} "
                              f"(attn={attn_implementation})")
                        return model
                    errors.append(
                        f"{name}: built {type(model).__name__} but it is missing "
                        f"{self.layer_names_dict['layer_prefix']}"
                    )
                    continue
                print(f"built empty {type(model).__name__} via {name} (attn={attn_implementation})")
                return model
        raise RuntimeError(
            "Could not instantiate the model on meta from config. "
            f"architecture={getattr(self.config, 'architectures', None)} "
            f"model_type={getattr(self.config, 'model_type', None)}. "
            f"Tried: {'; '.join(errors)}"
        )

    def init_model(self):
        # Build the real model on meta (no memory). include_buffers=False so non-persistent
        # buffers such as rotary inv_freq are actually computed (they aren't in the checkpoint).
        try:
            self.model = self._instantiate_on_meta("sdpa")
        except Exception as e:
            print(f"attn_implementation='sdpa' not available ({e}), falling back to eager attention")
            # Some (often remote-code) architectures don't support sdpa and also default to it, so we
            # must request eager explicitly; otherwise transformers re-selects sdpa and errors again.
            self.model = self._instantiate_on_meta("eager")

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is None:
            # Nested multimodal configs (Kimi K3) keep it under text_config.
            quantization_config = getattr(getattr(self.config, "text_config", None),
                                          "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
            # compressed-tensors registers a hook that expands every packed module on the first
            # forward. That undoes per-expert streaming (a K3 layer becomes ~56GB) and is also
            # unnecessary: we decompress each expert ourselves as it loads. Remove the hook.
            hook = getattr(self.model, "ct_decompress_hook", None)
            if hook is not None:
                hook.remove()
                delattr(self.model, "ct_decompress_hook")

        self.model.eval()
        self.model.tie_weights()
        self.model.generation_config = self.generation_config

        # Move all (already-materialized) buffers to the running device, preserving their dtype.
        # This includes rotary inv_freq, which transformers computes once at the model level and
        # passes down to every decoder layer.
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != 'meta':
                set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer)

        # Force the model to report the running (cuda) device even though its parameters live on
        # meta between layer executions, so transformers' generation utilities place inputs/cache
        # tensors on the right device.
        self._patch_device_property()

    def _patch_device_property(self):
        running_device = torch.device(self.running_device)
        running_dtype = self.running_dtype
        base_cls = type(self.model)

        # transformers >= 4.50 removed GenerationMixin from PreTrainedModel, so model classes that
        # predate that change (or ship as remote code, like Kimi K3's multimodal wrapper) no longer
        # have .generate(). They still define prepare_inputs_for_generation, so mixing the class
        # back in restores generation. It must come after the model class, per transformers.
        extra_bases = () if isinstance(self.model, GenerationMixin) else (GenerationMixin,)
        if extra_bases:
            print(f"{base_cls.__name__} does not inherit GenerationMixin; mixing it in so "
                  f"generate() works.")

        class _AirLLMRuntimeModel(base_cls, *extra_bases):
            @property
            def device(self):
                return running_device

            @property
            def dtype(self):
                return running_dtype

        self.model.__class__ = _AirLLMRuntimeModel

    def _module_at(self, dotted_name):
        module = self.model
        for attr_name in dotted_name.split('.'):
            module = getattr(module, attr_name)
        return module

    def set_layers_from_layer_names(self):
        """Resolve ``layer_names`` to the module objects the hooks attach to.

        Built by walking ``layer_names`` in order rather than by hard-coding the four stages, so
        extra stages (a trunk-level post-embedding norm, a discovered position table) and stage
        slots that are intentionally empty stay index-aligned with the checkpoint shards.
        """
        self.layers = []
        block_prefix = self.layer_names_dict["layer_prefix"] + "."
        norm_name = self.layer_names_dict["norm"]
        for name in self.layer_names:
            if name.startswith(block_prefix) and name[len(block_prefix):].isdigit():
                blocks = self._module_at(self.layer_names_dict["layer_prefix"])
                self.layers.append(blocks[int(name[len(block_prefix):])])
                continue
            try:
                module = self._module_at(name)
            except AttributeError:
                if name == norm_name:
                    # A layout slot with no module of its own (a trunk-level norm kept as its own
                    # stage, a "stateless" placeholder). Point it at a parameterless module so the
                    # hook sequence still lines up with ``layer_names``.
                    module = torch.nn.Identity()
                    setattr(self, '_placeholder_norm_module', module)
                else:
                    raise
            self.layers.append(module)

    # ---- weight streaming -------------------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        t = time.time()
        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        if self.prefetching and torch.cuda.is_available():
            # pin_memory() returns a pinned copy rather than pinning in place, so the result has to
            # be kept for the faster host->device copy to actually happen. Pinned memory can't be
            # paged out, so we only spend it on layers small enough to be safe: a frontier MoE
            # checkpoint has ~17GB layers, and with prefetching two are in flight at once, which
            # would lock up ~34GB of RAM for a copy that is dwarfed by the disk read anyway.
            total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
            if total_bytes <= self.max_pinned_layer_bytes:
                try:
                    for k in state_dict.keys():
                        state_dict[k] = state_dict[k].pin_memory()
                except RuntimeError:
                    # Out of pinned memory: fall back to pageable, which is slower but always works.
                    pass

        return state_dict

    def _restore_plain_weight_modules(self, state_dict):
        if self.hf_quantizer is None:
            # Nothing ever packed these modules, so there is no layout to undo. This is the
            # overwhelmingly common case and it runs once per streamed module, so skip the scan.
            return
        """Undo CT's packed-parameter layout when the checkpoint still ships a plain ``weight``.

        Some checkpoints (Kimi K3) list residual/router Linears under the MXFP4 target list but
        store them as bf16. After ``preprocess_model`` those modules expose ``weight_packed`` and
        reject the real ``weight`` tensor. Restore a meta ``weight`` Parameter so the shard can load.
        """
        plain = {k[:-len('.weight')] for k in state_dict if k.endswith('.weight')}
        packed = {k[:-len('.weight_packed')] for k in state_dict if k.endswith('.weight_packed')}
        for prefix in plain - packed:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            names = list(module._parameters.keys())
            if 'weight' in names or 'weight_packed' not in names:
                continue
            weight = state_dict[f'{prefix}.weight']
            for name in names:
                if name == 'weight' or name.startswith('weight_'):
                    module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
            for attr in ('quantization_scheme', 'quantization_status', 'quantization_format'):
                if hasattr(module, attr):
                    delattr(module, attr)

    def _decompress_state_dict(self, state_dict):
        """Expand packed payloads into the plain weights the modules expect.

        compressed-tensors has no quantized compute kernels: it registers a hook that decompresses
        the whole model before the first forward, after which every quantized module wants a plain
        ``weight``. Our shards still hold ``weight_packed``/``weight_scale``, so we expand them here.

        The expansion happens on the GPU, after transferring the *packed* bytes. For MXFP4 that
        moves 4x less data across PCIe than transferring an already-expanded weight would.
        """
        if self.hf_quantizer is None:
            return state_dict

        packed_prefixes = {k[: -len('.weight_packed')]
                           for k in state_dict if k.endswith('.weight_packed')}
        if not packed_prefixes:
            return state_dict

        from compressed_tensors.compressors.base import BaseCompressor
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
        except ImportError:
            CompressedLinear = None

        out = dict(state_dict)
        for prefix in packed_prefixes:
            try:
                module = self.model.get_submodule(prefix)
            except AttributeError:
                continue
            # CompressedLinear's own forward consumes the packed payload, so leave it packed.
            if CompressedLinear is not None and isinstance(module, CompressedLinear):
                continue
            scheme = getattr(module, 'quantization_scheme', None)
            if scheme is None or getattr(scheme, 'format', None) is None:
                continue

            local = {k[len(prefix) + 1:]: v.to(self.running_device)
                     for k, v in state_dict.items() if k.startswith(prefix + '.')}
            # scheme.format is an enum on some compressed-tensors versions and a plain str on others.
            fmt = getattr(scheme.format, 'value', scheme.format)
            compressor = BaseCompressor.get_value_from_registry(fmt)
            decompressed = compressor.decompress(local, scheme)

            for k in local:
                out.pop(f'{prefix}.{k}', None)
            if 'weight' in decompressed:
                # Once expanded, the module reads only `weight`; the scales that came back merely
                # describe how the checkpoint stored it, and keeping them would waste VRAM.
                out[f'{prefix}.weight'] = decompressed['weight']
                self._expose_plain_weight(module, decompressed['weight'])
            else:
                for k, v in decompressed.items():
                    out[f'{prefix}.{k}'] = v
        return out

    def _expose_plain_weight(self, module, weight):
        """Swap a module's packed parameters for the plain ``weight`` its forward reads.

        compressed-tensors patches a ``quantized_forward`` onto quantized Linears that reads
        ``self.weight``; the packed parameters only describe how the checkpoint stores the value.
        Marking the module COMPRESSED tells that forward the weight is already on the quantization
        grid, so it skips a fake-quantize that would only reproduce what we just decompressed.
        """
        existing = module._parameters.get('weight')
        if existing is None or existing.shape != weight.shape:
            for name in [n for n in list(module._parameters) if n.startswith('weight')]:
                module._parameters.pop(name, None)
            module.register_parameter(
                'weight',
                torch.nn.Parameter(torch.empty(weight.shape, device='meta', dtype=weight.dtype),
                                   requires_grad=False),
            )
        try:
            from compressed_tensors.quantization import QuantizationStatus
        except ImportError:
            return
        module.quantization_status = QuantizationStatus.COMPRESSED

    def _adopt_checkpoint_shape(self, param_name, value):
        # Fast path: once a parameter has been seen with the shape the model builds, this is a
        # dict hit instead of a getattr walk. Shapes are per-checkpoint constants.
        agreed = getattr(self, '_shape_agreed', None)
        if agreed is not None and param_name in agreed:
            return
        """Resize a meta placeholder whose shape disagrees with the checkpoint.

        A model class builds its parameters from the config, and that construction can disagree
        with the weights actually shipped. Kimi K3 sizes ``A_log`` from ``num_heads`` while its
        checkpoint stores one entry per head channel, which would abort the load. For a parameter
        still on meta -- i.e. one we have never materialised -- the checkpoint is the source of
        truth, so adopt its shape.
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            owner = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            return
        param = owner._parameters.get(attr)
        if param is None or param.device.type != 'meta' or param.shape == value.shape:
            return
        if agreed is None:
            agreed = self._shape_agreed = {}
        agreed[param_name] = value.shape
        param.data = torch.empty(value.shape, dtype=param.dtype, device='meta')

    def _place_via_accelerate(self, param_name, value, verbatim):
        """Slow placement path, used whenever the storage cannot be reused in place.

        ``set_module_tensor_to_device`` keeps the *existing* parameter dtype when ``dtype`` is not
        given, so a packed int8 payload handed to a float parameter silently comes back as floats --
        the quantization is destroyed and the values turn into noise. Passing the payload's own dtype
        through for verbatim tensors is what keeps pre-quantized checkpoints intact.

        Integer payloads need one more step: ``nn.Parameter`` cannot be created with
        ``requires_grad=True`` from a non-float tensor, so accelerate raises before it ever places
        the value. Those become a plain buffer instead -- which is what they represent anyway.
        """
        if verbatim and value.device.type != 'meta' and not value.is_floating_point():
            self._place_integer_payload(param_name, value)
            return
        kwargs = {}
        if not verbatim:
            kwargs['dtype'] = self.running_dtype
        elif value.device.type != 'meta':
            kwargs['dtype'] = value.dtype
        set_module_tensor_to_device(self.model, param_name, self.running_device,
                                    value=value, **kwargs)

    def _place_integer_payload(self, param_name, value):
        """Attach an integer tensor (packed 4-bit weights, zero points, indices) to a module.

        These cannot be ``nn.Parameter``: torch refuses to build a grad-requiring Parameter from a
        non-float tensor. A non-persistent buffer holds them exactly as loaded, keeps them in
        ``state_dict`` lookups via the module, and stays out of the checkpoint we would re-save.
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            owner = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            owner = None
        if owner is None:
            raise ValueError(f"cannot place {param_name}: no such module on the model")
        owner._parameters.pop(attr, None)
        owner._buffers.pop(attr, None)
        owner.register_buffer(attr, value.to(self.running_device), persistent=False)

    def _place_plain_tensor(self, param_name, value):
        """Place an ordinary high-precision tensor on the running device.

        ``set_module_tensor_to_device`` does a lot of work we do not need here: it re-validates
        shapes, rebuilds ``Parameter`` subclasses (which costs a ``_make_subclass`` per tensor) and
        re-reads the module tree. For the overwhelmingly common case -- a plain float tensor going
        to the device the module already lives on -- swapping the storage in place is equivalent
        and roughly 10x cheaper. That matters because this runs once per weight per layer per
        generated token.

        Falls back to accelerate whenever anything is non-standard (a dtype change, a quantized
        Parameter subclass, a size mismatch, or a device we cannot cheaply assert).
        """
        module_path, _, attr = param_name.rpartition('.')
        try:
            owner = self.model.get_submodule(module_path) if module_path else self.model
        except AttributeError:
            owner = None
        param = owner._parameters.get(attr) if owner is not None else None

        # A pre-quantized payload must be placed exactly as it arrived; an ordinary tensor must end
        # up as ``running_dtype``. Both conditions have to hold before we can reuse the storage.
        verbatim = self._should_load_verbatim(param_name, value)
        wanted_dtype = value.dtype if verbatim else self.running_dtype
        can_reuse = (
            param is not None
            and type(param) is torch.nn.Parameter
            and param.device.type == value.device.type
            and (param.device.type == 'cpu' or param.device.index == value.device.index)
            and param.shape == value.shape
            and value.dtype == wanted_dtype
            and (value.device.type == 'meta' or param.dtype == wanted_dtype)
        )
        if not can_reuse:
            self._place_via_accelerate(param_name, value, verbatim)
            return

        # Keep the Parameter object (tie_weights and state_dict keys depend on it) and just point
        # it at the new data. The previous storage is released when the old tensor is collected.
        param.data = value
        if value.device.type != 'meta' and param.grad is not None:
            param.grad = None

    def move_layer_to_device(self, state_dict):
        self._restore_plain_weight_modules(state_dict)
        state_dict = self._decompress_state_dict(state_dict)
        marker = self.layer_names_dict.get('cpu_resident_marker')
        moved = []
        for param_name in self._param_names_from_state_dict(state_dict):
            # A nested CPU table (Flash-Next PLE) must never be placed on the GPU, even if a
            # decoder-layer shard still contains it.
            if param_name in getattr(self, '_cpu_resident_params', ()):
                continue
            if marker and marker in param_name:
                continue
            if self.hf_quantizer is not None and self._needs_quantization(param_name):
                # On-the-fly-quantizing schemes (e.g. bitsandbytes) reconstruct the param from the
                # weight plus companion quant-state tensors carried in state_dict.
                self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name,
                                                         self.running_device, state_dict)
            else:
                # Normal load. Only ordinary high-precision tensors get cast to the runtime dtype;
                # pre-quantized payloads must be placed verbatim (see _should_load_verbatim).
                value = state_dict[param_name]
                self._adopt_checkpoint_shape(param_name, value)
                self._place_plain_tensor(param_name, value)
            moved.append(param_name)
        return moved

    # Suffixes of the companion tensors that pre-quantized checkpoints ship alongside a weight
    # (fp8 block scales, compressed-tensors/MXFP4 packed payloads and their scales, GPTQ indices).
    _QUANT_COMPANION_SUFFIXES = ("_scale", "_scale_inv", "_packed", "_zero_point", "_g_idx", "_shape")

    def _should_load_verbatim(self, param_name, value):
        """Whether a checkpoint tensor must be placed on the device without a dtype cast.

        Casting a pre-quantized payload to the runtime dtype destroys it: an fp8 weight loses its
        quantization, and a 4-bit MXFP4 ``weight_packed`` tensor (stored as packed integers) becomes
        meaningless floats. Equally important for very large models, decompressing on load would
        multiply a layer's footprint by ~4x, which is what keeps Kimi-K3-class checkpoints from
        fitting on a single GPU. So we keep anything that isn't a plain high-precision float as-is.
        """
        if not value.is_floating_point():
            # Packed 4-bit payloads, zero points, g_idx, shape metadata.
            return True
        if value.element_size() == 1:
            # Any 8-bit float: fp8 e4m3/e5m2 weights, and the e8m0 scales MXFP4 uses.
            return True
        return param_name.endswith(self._QUANT_COMPANION_SUFFIXES)

    def _needs_quantization(self, param_name):
        q = self.hf_quantizer
        # transformers renamed check_quantized_param -> param_needs_quantization.
        if hasattr(q, "param_needs_quantization"):
            return q.param_needs_quantization(self.model, param_name)
        return q.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})

    def _param_names_from_state_dict(self, state_dict):
        names = []
        for param_name in state_dict.keys():
            # bitsandbytes stores a weight plus companion quant-state tensors named
            # "<weight>.4bit.*" / "<weight>.8bit.*"; those are reconstructed together via
            # create_quantized_param, so collapse them down to the base weight name. Everything
            # else (including fp8 weight + weight_scale_inv pairs) is kept as distinct params.
            if '.4bit.' in param_name or '.8bit.' in param_name:
                base = param_name.split('.4bit.')[0].split('.8bit.')[0]
                if base not in names:
                    names.append(base)
            elif param_name not in names:
                names.append(param_name)
        return names

    def _load_resident_modules(self):
        """Load modules that sit outside the streamed embed -> layers -> norm -> lm_head sequence.

        Multimodal checkpoints carry a vision tower and projector, some architectures add extra
        top-level norms, and GPT-2 keeps a learned position table beside its token embedding.
        They never get a streaming hook, so without this they would stay on the meta device and
        fail the moment they run. They are small, so we load them once and leave them resident.
        """
        for name in self.layer_names_dict.get('resident', []):
            try:
                state_dict = self.load_layer_to_cpu(name)
            except FileNotFoundError:
                # Not every checkpoint of a given architecture ships every optional module.
                # A leaf tensor (rather than a module) is stored inside its parent's shard, so
                # fall back to reading just that tensor.
                state_dict = self._load_leaf_tensor(name)
                if state_dict is None:
                    continue
            except Exception:  # noqa: BLE001 - a resident module must never block construction
                continue
            self.move_layer_to_device(state_dict)

    def _load_leaf_tensor(self, name):
        """Read a single non-module tensor (e.g. ``transformer.wpe``) out of the split shards."""
        for idx in range(len(self.layer_names)):
            try:
                names = layer_tensor_names(self.checkpoint_path, self.layer_names[idx])
            except Exception:  # noqa: BLE001
                continue
            if name in names:
                return load_layer_subset(self.checkpoint_path, self.layer_names[idx], [name])
        return None

    def _cpu_resident_names(self):
        """Module prefixes whose weights stay on CPU for the lifetime of the process.

        Flash-Next's n-gram table is the motivating case: transformers already gathers rows onto
        the embedding's own device, so keeping ~102GB of bf16 on the host is the designed path.
        Names come from ``cpu_resident`` and from split files matching ``cpu_resident_marker``.
        """
        names = list(self.layer_names_dict.get('cpu_resident', []))
        marker = self.layer_names_dict.get('cpu_resident_marker')
        if marker:
            for path in Path(self.checkpoint_path).glob('*.mmap.json'):
                stem = path.name[:-len('.mmap.json')]
                if stem.endswith(marker) and stem not in names:
                    names.append(stem)
            for path in Path(self.checkpoint_path).glob('*.safetensors'):
                stem = path.name[:-len('.safetensors')]
                if stem.endswith(marker) and stem not in names:
                    names.append(stem)
        return names

    def _install_mmap_embedding(self, module_name, table):
        parent_name, _, attr = module_name.rpartition('.')
        parent = self.model.get_submodule(parent_name)
        setattr(parent, attr, MmapEmbedding(table))

    def _load_cpu_resident_modules(self):
        """Load oversized lookup tables onto CPU and never stream them to the GPU.

        Flash-Next's n-gram table is ~102GB bf16. We mmap it from disk so a 64GB host can still
        run; transformers already gathers rows on ``ngram_embedding.weight.device``.
        ``module.to('meta')`` after a decoder layer would otherwise evict a child embedding that
        lives under that layer, so we also record the parameter names and evict selectively.
        """
        self._cpu_resident_params = set()
        for name in self._cpu_resident_names():
            mmap_meta = Path(self.checkpoint_path) / f"{name}.mmap.json"
            if mmap_meta.exists():
                table = open_ngram_mmap_table(self.checkpoint_path, name)
                self._install_mmap_embedding(name, table)
                self._cpu_resident_params.add(f"{name}.weight")
                print(f"cpu-resident mmap embedding: {name} "
                      f"{tuple(table.shape)} {table.dtype} (gathered from disk, not copied to RAM)")
                continue
            try:
                state_dict = load_merged_ngram_embedding(self.checkpoint_path, name)
            except FileNotFoundError:
                continue
            for param_name, value in state_dict.items():
                self._adopt_checkpoint_shape(param_name, value)
                if self._should_load_verbatim(param_name, value):
                    set_module_tensor_to_device(self.model, param_name, 'cpu', value=value)
                else:
                    set_module_tensor_to_device(
                        self.model, param_name, 'cpu', value=value, dtype=self.running_dtype)
                self._cpu_resident_params.add(param_name)
        if self._cpu_resident_params:
            n = len(self._cpu_resident_params)
            print(f"cpu-resident modules: left {n} tensors on host RAM "
                  f"(n-gram / PLE table is gathered on CPU, not streamed to the GPU).")

    def _install_streaming_hooks(self):
        # Modules execute in this order during a forward: embed -> layers -> norm -> lm_head.
        n = len(self.layer_names)

        # Detect tied input/output embeddings. When tied, lm_head shares the embedding weight, so
        # there is no separate lm_head shard. We keep the embedding resident on the GPU (it is the
        # only copy and such models are small) and re-tie lm_head to it, then stream only the
        # decoder layers and the final norm.
        self.tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", False))

        if self.tie_word_embeddings:
            embed_state = self.load_layer_to_cpu(self.layer_names[0])
            self.move_layer_to_device(embed_state)
            self.model.tie_weights()
            self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
        else:
            self._streamed_indices = list(range(n))

        self._streamed_set = set(self._streamed_indices)

        # Checked before any hook is installed: a streamed module must never contain a
        # non-streamed sub-model (a vision tower would be evicted after its first forward).
        self._stream_root_guard()
        self._setup_expert_streaming()
        self._setup_layer_cache()

        if self._maybe_full_load():
            return

        for idx in self._streamed_indices:
            module = self.layers[idx]
            module._airllm_idx = idx
            # The checkpoint-level name this module owns. Parameter lookups need it: a streamed
            # module receives names like ``model.layers.7.mlp...`` while its own parameters are
            # named relative to itself (``mlp....``), and the index into ``layers`` is *not* the
            # index in the layer name.
            module._airllm_name = self.layer_names[idx]
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    # ---- full load --------------------------------------------------------------------------

    def _maybe_full_load(self):
        """Resident-load everything when it fits, instead of streaming one module at a time.

        Streaming trades speed for memory: when the whole model fits, that trade is pure loss.
        This is checked against the memory actually free at start-up, so it never turns a
        workable streaming setup into an OOM: any doubt (no probe, an oversized vision tower, an
        exotic quantizer) keeps the streaming path, which is the one that always runs.
        """
        if self.full_load is False or self.hf_quantizer is not None:
            return False
        if self.tie_word_embeddings:
            # The embed shard is resident already and lm_head shares it; there is nothing to
            # stream but the decoder layers, and the ordinary path handles that correctly.
            pass

        total = 0
        for idx in self._streamed_indices:
            total += self._layer_nbytes(idx)
        for name in self.layer_names_dict.get('resident', []):
            try:
                total += (Path(self.checkpoint_path) / (name + '.safetensors')).stat().st_size
            except OSError:
                pass

        from .utils import free_device_bytes
        free = free_device_bytes(self.running_device)
        if self.full_load is None:
            if free is None:
                return False
            # Require a comfortable margin: activations and the KV cache grow with context.
            if total * 1.25 > free:
                return False
        elif free is not None and total > free:
            print(f"full_load requested but {total / 1024 ** 3:.1f}GB of weights do not fit in "
                  f"{free / 1024 ** 3:.1f}GB free; streaming instead.")
            return False

        try:
            for idx in self._streamed_indices:
                state_dict = self._load_streamed_layer(idx)
                self.layers[idx]._airllm_moved = self.move_layer_to_device(state_dict)
                del state_dict
            self._load_resident_modules()
        except Exception as e:  # noqa: BLE001 - fall back to the path that always works
            print(f"full load failed ({type(e).__name__}: {e}); falling back to streaming.")
            self.clear_layer_cache()
            return False

        self.full_loaded = True
        # Every module stays materialised, so the layer cache and prefetch machinery are moot.
        self.prefetching = False
        self._streamed_set = set()
        self.reporter.begin('full load', f"{total / 1024 ** 3:.2f}GB resident")
        self.reporter.end('full load')
        where = f"{free / 1024 ** 3:.1f}GB free" if free is not None else "the requested device"
        print(f"full load: the whole {total / 1024 ** 3:.2f}GB checkpoint fits in {where}; "
              f"running without layer streaming.")
        return True

    # ---- layer cache ------------------------------------------------------------------------

    def _estimate_layer_bytes(self, idx):
        """Rough byte size of a streamed module, used to size the cache.

        The shard file is the honest number on the non-expert path: it is exactly what a reload
        reads. When per-expert streaming is on (or the file size is unavailable) fall back to the
        meta parameters, which are still shaped like the real weights at this point.
        """
        if self._expert_streaming and idx in self._non_expert_keys:
            # The streamed part of a MoE layer is everything except the experts; measuring the
            # file would count the (much larger) expert payload we do not load here.
            keys = self._non_expert_keys[idx]
            module = self.layers[idx]
            total = 0
            for name in keys:
                module_path, _, attr = name.rpartition('.')
                try:
                    owner = module.get_submodule(module_path) if module_path else module
                except AttributeError:
                    continue
                param = owner._parameters.get(attr) or owner._buffers.get(attr)
                if param is not None and param.device.type == 'meta':
                    continue
                if param is not None:
                    total += param.numel() * param.element_size()
            return total

        path = Path(self.checkpoint_path) / (self.layer_names[idx] + '.safetensors')
        try:
            size = path.stat().st_size
            if size:
                return size
        except OSError:
            pass
        total = 0
        for param in self.layers[idx].parameters(recurse=False):
            total += param.numel() * param.element_size()
        return total

    def _setup_layer_cache(self):
        """Size the two cache tiers and make sure one whole layer always fits.

        The VRAM tier is budgeted so that holding cached layers *plus* the layer currently being
        loaded never exceeds a fraction of what is free right now. That fraction is the safety
        margin that keeps a tight `--max-vram` run (or a card shared with the desktop) from
        OOMing: caching is an optimization, and must never be the reason a model stops running.
        """
        from .utils import free_device_bytes, total_host_bytes

        sample_idxs = [i for i in self._streamed_indices
                       if not (self._expert_streaming and i in self._expert_keys)]
        sizes = {}
        for idx in sample_idxs:
            sizes[idx] = self._estimate_layer_bytes(idx)
        self._layer_bytes = sizes

        # ---- VRAM tier ----
        if self.layer_cache_gb is None:
            free = free_device_bytes(self.running_device)
            if free is None:
                # CPU / MPS: no cheap "free memory" probe, and on CPU streaming is already a
                # host->host copy. Leave the VRAM tier off.
                gpu_budget = 0
            else:
                # 30% of free memory, but always leave room for the incoming layer itself.
                headroom = max((sizes.get(i, 0) for i in sample_idxs), default=0)
                gpu_budget = int(max(0, free * 0.30 - headroom))
        else:
            gpu_budget = int(max(0.0, float(self.layer_cache_gb)) * 1024 ** 3)

        # ---- pinned-host tier ----
        if self.park_cache_gb is None:
            host = total_host_bytes()
            if host is None or self.park_cache_gb == 0:
                park_budget = 0
            else:
                # Pinned memory cannot be swapped, so stay modest: 6% of RAM, capped at 8GB.
                park_budget = int(min(host * 0.06, 8 * 1024 ** 3))
        else:
            park_budget = int(max(0.0, float(self.park_cache_gb)) * 1024 ** 3)

        # Pinning only pays for itself when the copy crosses PCIe. On other accelerators (MPS) the
        # "pinned" copy is not cheaper, so the tier is dropped rather than silently paying for a
        # second host copy. On CPU itself the parked copy still skips the safetensors read, which
        # is the same win as on CUDA minus the DMA speedup.
        if not torch.cuda.is_available() and self.device.type not in ('cpu',):
            park_budget = 0

        self.layer_cache = StreamedLayerCache(gpu_budget=gpu_budget, park_budget=park_budget)
        self._replay_cache_plan()

        # Prefetching compresses the pipeline to N layers in flight; scale the safety margin so
        # the extra in-flight layers are accounted for rather than silently overcommitting.
        # Default 2, not 1: see the prefetch_depth docstring -- a depth of one leaves the reader
        # no lookahead, so every read is awaited rather than overlapped.
        self._prefetch_depth = max(0, int(self.prefetch_depth if self.prefetch_depth is not None else 2))
        if self.prefetching and self._executor is not None:
            self._executor = ThreadPoolExecutor(max_workers=max(1, self._prefetch_depth))
        self._prefetch_futures = {}
        self._prefetch_started = set()

        largest = max(sizes.values(), default=0)
        if self.layer_cache.enabled:
            print(f"layer cache: {_format_bytes(gpu_budget)} VRAM tier, "
                  f"{_format_bytes(park_budget)} pinned-host tier "
                  f"(largest streamed layer {_format_bytes(largest)})")
            if largest and gpu_budget + park_budget < largest:
                # Silently caching nothing looks exactly like the cache being broken. The user
                # asked for it, so say why it cannot hold anything.
                print(f"  warning: both tiers together hold less than one layer "
                      f"({_format_bytes(largest)}); nothing can be cached. Raise "
                      f"layer_cache_gb / park_cache_gb, or leave them unset for auto sizing.")
        else:
            print("layer cache disabled: every decoded token will re-read the whole checkpoint.")

    # ---- per-expert streaming ---------------------------------------------------------------


    def _setup_expert_streaming(self):
        """Stream individual MoE experts instead of whole decoder layers, where that is possible.

        A sparse MoE layer holds hundreds of experts but routes each token to a handful of them.
        Materialising the whole layer is therefore enormously wasteful: for Kimi K3 a layer's
        experts are ~55GB expanded, of which a token touches ~1GB. Because the model calls each
        selected expert as its own module (and skips unselected ones), a forward hook per expert
        loads exactly the experts that actually run.

        This needs `expert_prefix` in layer_names_dict and safetensors shards, since it relies on
        reading individual tensors out of a shard.
        """
        self._expert_streaming = False
        self._expert_keys = {}
        self._non_expert_keys = {}

        expert_prefix = self.layer_names_dict.get('expert_prefix')
        if not expert_prefix:
            return
        if self.layer_names_dict.get('expert_layout') == 'packed':
            # Detection found a sparse model, but this layout stores every expert as a slice of one
            # 3-D tensor rather than as per-expert modules. The routing index is consumed inside a
            # single op, so no hook can tell which slice a token will read; selective loading would
            # mean patching the model's forward. Whole-layer streaming is correct, and saying so is
            # more useful than silently doing nothing.
            params = []
            try:
                node = self.layers[self._streamed_indices[0]]
                for attr in expert_prefix.split('.'):
                    node = getattr(node, attr)
                params = [tuple(p.shape) for _, p in node.named_parameters()]
            except (AttributeError, IndexError):
                pass
            shape = f" (e.g. {params[0]})" if params else ""
            print(f"sparse MoE detected, but experts are slices of one batched tensor{shape} "
                  f"rather than separate modules, so they stream as part of the whole layer. "
                  f"Only the routed experts' slice is computed, but the whole layer is read.")
            return
        if type(ModelPersister.get_model_persister()).__name__ != 'SafetensorModelPersister':
            return
        if self.compression is not None:
            # Per-expert loading reads tensors straight out of the shard with load_layer_subset(),
            # which -- unlike load_layer() -- does not run uncompress_layer_state_dict(). With
            # compression on, the shard holds a quantized payload plus companion .4bit./.8bit.
            # quant-state tensors, so streaming subsets of it would hand still-quantized weights to
            # move_layer_to_device(). Fall back to whole-layer streaming, which does decompress.
            print("per-expert streaming is not supported together with compression for now; "
                  "streaming whole layers instead.")
            return

        layer_prefix = self.layer_names_dict['layer_prefix']
        hooked = 0

        for idx in self._streamed_indices:
            layer_name = self.layer_names[idx]
            if not layer_name.startswith(layer_prefix + '.'):
                continue
            try:
                names = layer_tensor_names(self.checkpoint_path, layer_name)
            except Exception:
                continue

            marker = f'.{expert_prefix}.'
            per_expert = {}
            others = []
            for key in names:
                pos = key.find(marker)
                if pos == -1:
                    others.append(key)
                    continue
                rest = key[pos + len(marker):]
                head = rest.split('.', 1)[0]
                if not head.isdigit():
                    others.append(key)
                    continue
                per_expert.setdefault(int(head), []).append(key)

            if not per_expert:
                continue

            layer_module = self.layers[idx]
            experts_container = layer_module
            try:
                for attr in expert_prefix.split('.'):
                    experts_container = getattr(experts_container, attr)
            except AttributeError:
                continue

            self._non_expert_keys[idx] = others
            self._expert_keys[idx] = per_expert

            for expert_idx, keys in per_expert.items():
                if expert_idx >= len(experts_container):
                    continue
                expert_module = experts_container[expert_idx]
                expert_module._airllm_expert = (idx, expert_idx)
                expert_module.register_forward_pre_hook(self._expert_pre_hook)
                expert_module.register_forward_hook(self._expert_post_hook)
                hooked += 1

        if hooked:
            self._expert_streaming = True
            n_layers = len(self._expert_keys)
            print(f"per-expert streaming enabled: {hooked} experts across {n_layers} layers "
                  f"load on demand, so only the experts a token routes to are materialised.")

    def _expert_pre_hook(self, module, args):
        layer_idx, expert_idx = module._airllm_expert
        keys = self._expert_keys[layer_idx][expert_idx]
        state_dict = load_layer_subset(self.checkpoint_path, self.layer_names[layer_idx], keys)
        module._airllm_moved = self.move_layer_to_device(state_dict)

    def _expert_post_hook(self, module, args, output):
        for param_name in getattr(module, '_airllm_moved', []):
            set_module_tensor_to_device(self.model, param_name, 'meta')
        module._airllm_moved = []
        return output

    # ---- streaming hooks --------------------------------------------------------------------

    def _next_streamed_idx(self, idx):
        nxt = idx + 1
        return nxt if nxt in self._streamed_set else None

    def _layer_nbytes(self, idx):
        nbytes = self._layer_bytes.get(idx)
        if nbytes is None:
            nbytes = self._estimate_layer_bytes(idx)
            self._layer_bytes[idx] = nbytes
        return nbytes

    def _is_empty_module(self, idx):
        """True when a streaming slot has no parameters, so it has no shard to load."""
        if self.layer_names_dict.get('stateless_norm'):
            return True
        module = self.layers[idx] if idx < len(self.layers) else None
        if module is None:
            return True
        try:
            return next(module.parameters(), None) is None
        except AttributeError:
            return True

    def _load_streamed_layer(self, idx, decompress=True):
        """Load one streamed module's weights. Experts are excluded when they stream themselves.

        With compression on, ``decompress=False`` returns the still-quantized shard: that is what
        the prefetch thread fetches, so only the disk read is overlapped and only one expanded
        layer ever exists in memory.
        """
        if idx == len(self.layers) - 2 and self._is_empty_module(idx):
            # The norm slot exists in the sequence even when the architecture has nothing to put
            # there. GPT-2/Falcon have no trunk-level norm at all (``stateless_norm``); OLMo-style
            # models have a ``model.norm`` *module* with no parameters, so it is discovered as a
            # real stage while the splitter -- correctly -- writes no shard for it. Loading then
            # asked for a file that was never created. Treat "module exists but has no parameters"
            # the same as "no module": both mean there is nothing to stream.
            return {}
        keys = self._non_expert_keys.get(idx) if getattr(self, '_expert_streaming', False) else None
        if keys is None or not keys:
            if self.compression is not None:
                return load_layer(self.checkpoint_path, self.layer_names[idx],
                                  device=self.running_device, decompress=decompress)
            return self.load_layer_to_cpu(self.layer_names[idx])
        # ``keys`` are checkpoint-level names; load_layer_subset resolves them against the shard.
        state = load_layer_subset(self.checkpoint_path, self.layer_names[idx], keys)
        if self.compression is not None and decompress:
            state = uncompress_layer_state_dict(state, device=self.running_device)
        return state

    def _start_prefetch(self, idx):
        self._prefetch_started.add(idx)
        try:
            # Never decompress here: the caller materializes exactly one layer at a time.
            return self._load_streamed_layer(idx, decompress=self.compression is None)
        finally:
            self._prefetch_started.discard(idx)

    def _prefetch(self, idx):
        """Schedule a background read of ``idx`` unless it is already in flight or cached."""
        if not self.prefetching or self._executor is None:
            return
        if idx is None or idx in self._prefetch_futures:
            return
        if idx in self.layer_cache:
            return
        # Bound the queue: with a deep prefetch a slow disk would otherwise read the whole
        # checkpoint into host memory ahead of the compute that can consume it.
        while len(self._prefetch_futures) >= max(1, self._prefetch_depth):
            stalled, future = next(iter(self._prefetch_futures.items()))
            if stalled in self._prefetch_started:
                break  # already reading; let it finish rather than leaving a hole
            # Drop it *and* cancel it. Forgetting the entry without cancelling leaves a task that
            # still runs later and reads bytes nobody will consume -- the same duplicated read,
            # just harder to see.
            self._prefetch_futures.pop(stalled)
            future.cancel()
        self._prefetch_futures[idx] = self._executor.submit(self._start_prefetch, idx)

    def _take_prefetch(self, idx):
        """Consume the in-flight read for ``idx``, if any.

        ``Future.cancel`` is the only safe way to decide this, because it is the one operation
        that is atomic with respect to the worker: it returns True exactly when the task will
        never run. The previous version asked a bookkeeping set instead, and lost the race when a
        read had been picked up by a worker but had not yet recorded itself -- the read was then
        started a second time on this thread, so prefetching *doubled* the disk traffic and hid
        nothing. That is also why large prefetch depths measured slower than no prefetching.
        """
        future = self._prefetch_futures.pop(idx, None)
        if future is None:
            return None
        if future.cancel():
            # Never ran: read it here instead, it is the same disk either way.
            return None
        return future.result()

    def _prefetch_ahead(self, idx):
        """Queue the next ``prefetch_depth`` modules after ``idx``."""
        if not self.prefetching or self._executor is None:
            return
        nxt = idx
        for _ in range(max(1, self._prefetch_depth)):
            nxt = self._next_streamed_idx(nxt)
            if nxt is None:
                break
            self._prefetch(nxt)

    def _restore_parked(self, entry):
        """Copy a parked layer back onto the device, returning the moved parameter names."""
        moved = []
        for param_name, value in entry.parked.items():
            value = value.to(self.running_device, non_blocking=True)
            set_module_tensor_to_device(self.model, param_name, self.running_device, value=value)
            moved.append(param_name)
        return moved

    def _layer_hit(self, idx):
        """Materialise a cached module. Returns its moved parameter names, or None on a miss."""
        cache = self.layer_cache
        if not cache.enabled:
            return None
        entry = cache.get(idx)
        if entry is None:
            return None
        module = self.layers[idx]
        if entry.resident:
            module._airllm_moved = list(entry.moved)
            return module._airllm_moved
        if entry.parked:
            module._airllm_moved = self._restore_parked(entry)
            cache.mark_resident(idx, module._airllm_moved)
            return module._airllm_moved
        # A parked copy that was dropped (budget shrank, or it never fit): treat as a miss.
        cache.discard(idx, release=False, count_eviction=False)
        return None

    def _make_room(self, idx, nbytes, want_resident=True):
        """Evict whatever the cache has to give up before ``idx`` can be cached."""
        for entry in self.layer_cache.plan(nbytes, incoming_idx=idx,
                                          want_resident=want_resident):
            self._evict_entry(entry)

    def _param_tensor(self, param_name, module=None):
        """Resolve a parameter name to its tensor, accepting checkpoint- or module-level names.

        ``move_layer_to_device`` records checkpoint-level names (``model.layers.7.mlp.gate_proj``)
        while a hook only owns one module, whose own parameters are named relative to itself
        (``mlp.gate_proj.weight``). Pass the module that owns the weights and its own prefix is
        stripped; pass nothing and the name is resolved against the whole model.

        Getting this wrong is silent, not loud: the lookup returns None, the caller skips the
        parameter, and the layer is quietly dropped from the cache instead of kept.
        """
        if module is not None:
            candidates = []
            own = getattr(module, '_airllm_param_prefix', None)
            if own:
                candidates.append(own)
            # The layer's own name from the layout. ``layer_names`` holds the checkpoint-level name
            # (``model.layers.7``) for the module the hooks are attached to, and a streamed module's
            # index in that list is not its index in ``layers`` -- looking the prefix up by position
            # silently matches the wrong block, which is worse than not matching at all.
            name = getattr(module, '_airllm_name', None)
            if name:
                candidates.append(name + '.')
            for prefix in (self.layer_names_dict.get('layer_prefix'), 'transformer.h',
                           'gpt_neox.layers', 'model.decoder'):
                if prefix:
                    candidates.append(prefix + '.')
            for prefix in candidates:
                if not prefix or not param_name.startswith(prefix):
                    continue
                path, _, attr = param_name[len(prefix):].rpartition('.')
                try:
                    sub = module.get_submodule(path) if path else module
                except AttributeError:
                    continue
                tensor = sub._parameters.get(attr)
                if tensor is None:
                    tensor = sub._buffers.get(attr)
                if tensor is not None:
                    return tensor
            # Fall back to a name that is already relative to the module.
            path, _, attr = param_name.rpartition('.')
            try:
                sub = module.get_submodule(path) if path else module
            except AttributeError:
                sub = None
            if sub is not None:
                tensor = sub._parameters.get(attr)
                if tensor is None:
                    tensor = sub._buffers.get(attr)
                if tensor is not None:
                    return tensor
            return None

        path, _, attr = param_name.rpartition('.')
        try:
            sub = self.model.get_submodule(path) if path else self.model
        except AttributeError:
            return None
        tensor = sub._parameters.get(attr)
        if tensor is None:
            tensor = sub._buffers.get(attr)
        return tensor

    def _evict_entry(self, entry):
        """Send a cached module back to ``meta``, parking a pinned host copy when allowed."""
        module = self.layers[entry.idx]
        cache = self.layer_cache
        moved = list(getattr(module, '_airllm_moved', entry.moved) or ())
        if not moved:
            cache.discard(entry.idx, release=False)
            return
        # One quick host copy turns the next miss into a memcpy instead of a disk read.
        parked = self._parked_copy(module, moved, entry.nbytes)
        evict_module(module, moved)
        module._airllm_moved = []
        cache.mark_evicted(entry.idx, parked)

    def _park_or_discard(self, idx):
        """Post-forward release for a module that does not stay materialised.

        The module still holds its weights at this point (nothing evicted it yet), so this is
        where they go back to ``meta``. A resident entry is handed to :meth:`_evict_entry`, which
        keeps a pinned host copy when the budget allows. A module the cache never claimed is where
        the host-parking decision is actually made: it was loaded, it ran, nothing in VRAM wants to
        keep it, so offer it to the parked tier *now* -- that copy is the only thing standing
        between the next token and another disk read.
        """
        cache = self.layer_cache
        module = self.layers[idx]
        entry = cache.peek(idx) if cache.enabled else None
        if entry is not None:
            self._evict_entry(entry)
            return True
        moved = list(getattr(module, '_airllm_moved', None) or ())
        if not moved:
            module._airllm_moved = []
            return False
        if not cache.enabled:
            evict_module(module, moved)
            module._airllm_moved = []
            return False

        nbytes = self._layer_nbytes(idx)
        parked = self._parked_copy(module, moved, nbytes)
        evict_module(module, moved)
        module._airllm_moved = []
        if parked is not None:
            self._make_room(idx, nbytes, want_resident=False)
            cache.record(idx, nbytes, moved, resident=False, parked=parked,
                         priority=self._cache_priority(idx, nbytes))
            return False
        cache.discard(idx, release=False, count_eviction=False)
        return False

    def _parked_copy(self, module, moved, nbytes):
        """Snapshot a module's weights into host memory, or None if the budget says no.

        Prefers pinned memory (the copy then crosses PCIe without a bounce buffer) but falls back
        to a pageable *copy*: it must never alias the tensors being evicted, or holding the copy
        would keep the very storage the eviction is trying to free alive.
        """
        if not self.layer_cache.can_park(nbytes):
            return None
        parked = {}
        for param_name in moved:
            param = self._param_tensor(param_name, module)
            if param is None or param.device.type == 'meta':
                continue
            value = param.data
            try:
                value = value.pin_memory()
            except (RuntimeError, AssertionError):
                value = value.detach().clone()
            parked[param_name] = value
        return parked or None

    def _cache_priority(self, idx, nbytes):
        """How costly a miss on ``idx`` is, relative to its footprint.

        A decoder layer is a multi-GB disk read; the final norm is a few KB, so treating "one LRU
        entry" as one unit of value would let a 4KB norm evict a 2GB layer. Two signals feed the
        score:

        * cost of the reload: norms are trivial, the output head is a plain gather, layers are
          full shard reads;
        * how soon it runs again: the head of the model is re-read first on the next token, so
          when the budget cannot hold everything, caching the *start* of the stack means the next
          pass is immediately computing instead of immediately waiting on disk.
        """
        name = self.layer_names[idx]
        if name.endswith('norm') or name.endswith('layernorm'):
            return 0.1
        if name == self.layer_names_dict.get('lm_head'):
            return 0.75
        # Positional bonus, decaying from 1.25 at the head to 1.0 at the tail.
        try:
            position = self._streamed_indices.index(idx)
            span = max(1, len(self._streamed_indices) - 1)
            return 1.0 + 0.25 * (1.0 - position / span)
        except ValueError:
            return 1.0

    def _register_loaded(self, idx, nbytes):
        """Record a freshly loaded module, keeping it resident while the budget allows.

        A module that is about to run again soonest (the first layers of the next pass) should be
        the *last* thing evicted, and an LRU already gives us that for free: the module just
        loaded is the most recently used. So registration happens before any eviction for the
        following layer, and a tight budget naturally ends up caching the head of the model —
        which is exactly the part every token touches first.
        """
        cache = self.layer_cache
        module = self.layers[idx]
        moved = list(getattr(module, '_airllm_moved', []) or [])
        if not cache.enabled:
            return False
        priority = self._cache_priority(idx, nbytes)
        if cache.can_hold_resident(nbytes):
            self._make_room(idx, nbytes, want_resident=True)
            cache.record(idx, nbytes, moved, resident=True, priority=priority)
            return True
        if cache.can_park(nbytes):
            # Deliberately *not* recorded. Reserving the host budget here would mean registering an
            # entry that holds nothing: the weights are only copied once the module has run and is
            # being released (_park_or_discard), and until then the entry is a promise the cache
            # cannot distinguish from a real hit. Recording it now made the cache claim hits for
            # layers it had no copy of, and those layers were then re-read on every token.
            return False
        cache.discard(idx, release=False, count_eviction=False)
        return False

    def _pre_hook(self, module, args):
        idx = module._airllm_idx

        if idx == len(self.layers) - 2 and self._is_empty_module(idx):
            # A placeholder slot for an architecture with no final norm (or one whose norm module
            # holds no parameters): nothing to stream.
            module._airllm_moved = []
            return

        if self._layer_hit(idx) is not None:
            self._prefetch_ahead(idx)
            self.throughput.tick()
            return

        # A saved plan names the modules that paid off last run. On the first miss of a pass, kick
        # off every planned read that is still going to happen: these are exactly the layers that
        # are cheap to keep, so fetching them early means the pass converges to a fully cached
        # steady state sooner instead of re-deriving the working set one layer at a time.
        if self._planned_resident and not self._plan_primed:
            self._plan_primed = True
            for planned in self._planned_order:
                if planned > idx:
                    self._prefetch(planned)

        nbytes = self._layer_nbytes(idx)

        state_dict = self._take_prefetch(idx)
        if state_dict is None:
            state_dict = self._load_streamed_layer(idx)
        elif self.compression is not None:
            state_dict = uncompress_layer_state_dict(state_dict, device=self.running_device)

        module._airllm_moved = self.move_layer_to_device(state_dict)
        del state_dict
        self._register_loaded(idx, nbytes)
        self._prefetch_ahead(idx)
        self.throughput.tick()
        if self.reporter.enabled:
            rate = self.throughput.rate()
            self.reporter.beat(f"streaming layers at {rate:.1f}/s")

    def _post_hook(self, module, args, output):
        idx = module._airllm_idx
        entry = self.layer_cache.peek(idx) if self.layer_cache.enabled else None
        if entry is not None and entry.resident:
            # Keeping it loaded: that is the whole point of the cache.
            return output
        self._park_or_discard(idx)
        return output

    # ---- lifecycle --------------------------------------------------------------------------

    # ---- cache plan persistence -------------------------------------------------------------

    def _cache_plan_key(self):
        """Identifies the (checkpoint, device, budget) combination a plan was measured for."""
        device = str(self.device)
        try:
            index = self.device.index
            if index is not None and torch.cuda.is_available():
                device += f":{torch.cuda.get_device_properties(index).name}"
        except Exception:  # noqa: BLE001 - a plan key is cosmetic
            pass
        return {'checkpoint': str(self.checkpoint_path), 'device': device,
                'gpu_budget': self.layer_cache.gpu_budget}

    def _replay_cache_plan(self):
        """Apply a previously recorded resident set, if it still matches this run."""
        path = self.cache_plan_path
        if not path or not self.layer_cache.enabled:
            return
        try:
            plan = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return
        key = self._cache_plan_key()
        if any(plan.get(k) != v for k, v in key.items()):
            return
        names = set(plan.get('modules') or [])
        if not names:
            return
        # A plan is a *prefetch* hint, not a claim that anything is cached: the weights still have
        # to be read once before a layer can be served from memory. Recording entries here would
        # create exactly the empty entries the cache must never have, so the plan only primes the
        # read-ahead, which is what actually makes the first token cheaper.
        sizes = {idx: self._layer_nbytes(idx)
                 for idx, name in enumerate(self.layer_names) if name in names}
        capacity = self.layer_cache.gpu_budget + self.layer_cache.park_budget
        planned, total = [], 0
        for idx in sorted(sizes):
            if total + sizes[idx] > capacity:
                break
            planned.append(idx)
            total += sizes[idx]
        self._planned_resident = {self.layer_names[i] for i in planned}
        self._planned_order = planned
        self._plan_primed = False
        if planned:
            print(f"layer cache: saved plan warm-starts {len(planned)} modules "
                  f"({_format_bytes(total)})")

    def save_cache_plan(self, path=None):
        """Persist the modules currently held resident, for a faster next start."""
        path = path or self.cache_plan_path
        if not path or not self.layer_cache.enabled:
            return None
        modules = [self.layer_names[e.idx] for e in self.layer_cache._entries.values()
                   if e.resident]
        if self.layer_names[-1] not in modules:
            # The output head is what produces the first token of the next pass; streaming it
            # costs a full shard read for a few MB of useful compute.
            modules.append(self.layer_names[-1])
        payload = dict(self._cache_plan_key())
        payload['modules'] = modules
        try:
            Path(path).write_text(json.dumps(payload, indent=2))
        except OSError as e:  # noqa: BLE001 - persisting is best-effort
            print(f"could not write cache plan to {path}: {e}")
            return None
        return path

    def _stream_root_guard(self):
        """Refuse to stream a module that contains a non-streamed sub-model.

        The streaming hooks evict whatever a module owns after it runs. If a module's subtree also
        holds a ``resident`` or ``cpu_resident`` sub-model (a vision tower, an n-gram table), that
        sub-model would be dropped after its first forward and the next pass would crash on meta
        weights. ``layer_names_dict`` is meant to name leaf modules, but a wrong/unknown
        architecture layout should fail loudly at construction instead of mysteriously later.
        """
        protected = set(self.layer_names_dict.get('resident', []))
        for name in self._cpu_resident_names():
            protected.add(name)
            protected.add(name + '.weight')
        for idx in getattr(self, '_streamed_indices', []):
            streamed = self.layer_names[idx]
            for name in protected:
                if name == streamed or name.startswith(streamed + '.'):
                    raise ValueError(
                        f"module {streamed!r} would be streamed, but it is (or contains) the "
                        f"non-streamed module {name!r}. Add an explicit layer layout for this "
                        f"architecture (see airllm/airllm_kimi_k3.py) so the tower is not evicted."
                    )

    def clear_layer_cache(self):
        """Drop cached layers, returning their memory to the OS/driver.

        Call this between unrelated generations, or before handing the GPU to another process:
        while the cache is warm it holds real VRAM and pinned RAM by design.
        """
        for entry in list(self.layer_cache.clear()):
            if entry.resident:
                module = self.layers[entry.idx]
                evict_module(module, getattr(module, '_airllm_moved', None) or entry.moved)
                module._airllm_moved = []
        for future in self._prefetch_futures.values():
            try:
                future.result()
            except Exception:  # noqa: BLE001 - a failed prefetch is retried on demand
                pass
        self._prefetch_futures.clear()
        clean_memory()

    def layer_cache_stats(self):
        """Hit/miss counters for the layer cache, plus peak memory figures where available."""
        stats = self.layer_cache.stats()
        stats['layer_bytes_max_mb'] = max(self._layer_bytes.values(), default=0) / 1024 ** 2
        stats['prefetch_depth'] = getattr(self, '_prefetch_depth', 0)
        if torch.cuda.is_available():
            stats['peak_vram_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        return stats


    # ---- delegation to the underlying transformers model ------------------------------------

    def generate(self, *args, **kwargs):
        max_new = kwargs.get('max_new_tokens')
        user_streamer = kwargs.get('streamer')
        progress_streamer = None
        if self.reporter.enabled and user_streamer is None and BaseStreamer is not object:
            # Counting tokens is the one signal that matches what the user is waiting for. A
            # user-supplied streamer takes precedence: passing one would turn generate() into a
            # generator and change the return type, so it is only installed when there is none.
            progress_streamer = _ProgressStreamer(self.reporter, self.throughput)
            kwargs['streamer'] = progress_streamer

        self.throughput.reset()
        self.reporter.begin('generate', f"max_new_tokens={max_new}")
        start = time.time()
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            if self.reporter.enabled:
                stats = self.layer_cache_stats()
                tokens = progress_streamer.tokens if progress_streamer is not None else None
                elapsed = max(1e-9, time.time() - start)
                rate = f", {tokens / elapsed:.2f} tok/s" if tokens else ""
                self.reporter.end(
                    'generate',
                    f"{stats.get('hits', 0)} cache hits / {stats.get('misses', 0)} loads "
                    f"({stats['hit_rate'] * 100:.0f}% hit rate){rate}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
