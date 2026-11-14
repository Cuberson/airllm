![airllm_logo](https://github.com/lyogavin/airllm/blob/main/assets/airllm_logo_sm.png?v=3&raw=true)

[**Quickstart**](#quickstart) | 
[**Configurations**](#configurations) | 
[**MacOS**](#macos) | 
[**Example notebooks**](#example-python-notebook) | 
[**FAQ**](#faq)

**AirLLM** dramatically reduces inference memory usage, letting 70B large language models run on a single 4GB GPU card — without quantization, distillation, or pruning. You can even run **Kimi K3 (2.8T)** — the largest open-source model released to date — on **under 4GB**, **Qwen3.8-Flash-Next (125B)** on **6GB**, and **DeepSeek-V3 (671B)** on **~12GB**. We now also support training huge models on small VRAM: **Qwen3.8-Flash-Next (125B)** under **6GB**.

<a href="https://github.com/lyogavin/airllm/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/lyogavin/airllm?style=social)</a>
[![Downloads](https://static.pepy.tech/personalized-badge/airllm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/airllm)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/wechat-Anima-brightgreen?logo=wechat)](https://static.aicompose.cn/static/wecom_barcode.png?t=1671918938)
[![Discord](https://img.shields.io/discord/1175437549783760896?logo=discord&color=7289da
)](https://discord.gg/2xffU5sn)
[![PyPI - AirLLM](https://img.shields.io/pypi/format/airllm?logo=pypi&color=3571a3)
](https://pypi.org/project/airllm/)
[![Website](https://img.shields.io/website?up_message=blog&url=https%3A%2F%2Fmedium.com%2F%40lyo.gavin&logo=medium&color=black)](https://medium.com/@lyo.gavin)
[![Website](https://img.shields.io/badge/Gavin_Li-Blog-blue)](https://gavinliblog.com)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dgavinli%26type%3Dpatrons&style=flat)](https://patreon.com/gavinli)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/lyogavin?logo=GitHub&color=lightgray)](https://github.com/sponsors/lyogavin)

## AI Agents Recommendation:

* [Best AI Game Sprite Generator](https://godmodeai.co)

* [Best AI Facial Expression Editor](https://crazyfaceai.com)

## Updates
[2026/09] **Training** support: stream frozen weights one layer at a time and keep adapters on the GPU. **Qwen3.8-Flash-Next (125B)** trains under **6GB** (RTX 3060 Ti); **Qwen3.8-27B** trains in **~2GB** at seq 512. See [Training](#training).

[2026/08] **Qwen3.8-Flash-Next** support: Qwen's 125B MoE flagship (`Qwen4ExpForConditionalGeneration`) with a ~51B n-gram embedding runs in **5.95GB** of VRAM, measured end to end on one RTX 4090. The n-gram table is file-mapped on the host (a 64GB machine is enough); decoder layers stream. Needs a `transformers` build with in-tree `qwen4_exp` (`pip install git+https://github.com/huggingface/transformers.git` today) and ~360GB of checkpoint disk (`delete_original=True` reclaims the originals after the split).

[2026/08] **Qwen3.8-27B** support: Qwen's new dense VL (Gated DeltaNet + Gated Attention, native vision) runs in **3.33GB** of VRAM, measured end to end on one RTX 3090. Needs `transformers` 5.8+.

[2026/07] **Kimi K3 (2.8T)** support: the largest open-source model runs on a single card in **3.72GB** of VRAM, measured end to end on one RTX 6000 Ada. Per-expert streaming loads only the experts a token actually routes to. K3 brings three requirements of its own: `pip install compressed-tensors flash-attn` (its model code mandates flash attention regardless of what you request), a CUDA 12 build of torch, since no prebuilt flash-attn wheel exists for CUDA 13 yet, and `transformers` 4.56.x, as its remote code does not load on 5.x.

[2026/06] **v3.0**: FP8 model support + the latest models. Run **DeepSeek-V3 (671B) on ~12GB** and **Qwen3-235B on ~3GB**, plus Qwen3, Llama 3.x/4, DeepSeek V2/V3, Phi-4, Gemma and more — all through a single `AutoModel`.

[2024/08/20] v2.11.0: Support Qwen2.5

[2024/08/18] v2.10.1 Support CPU inference. Support non sharded models. Thanks @NavodPeiris for the great work! 

[2024/07/30] Support Llama3.1 **405B** ([example notebook](https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_llama3.1_405B.ipynb)). Support **8bit/4bit quantization**.

[2024/04/20] AirLLM supports Llama3 natively already. Run Llama3 70B on 4GB single GPU.

[2023/12/25] v2.8.2: Support MacOS running 70B large language models.

[2023/12/20] v2.7: Support AirLLMMixtral. 

[2023/12/20] v2.6: Added AutoModel, automatically detect model type, no need to provide model class to initialize model.

[2023/12/18] v2.5: added prefetching to overlap the model loading and compute. 10% speed improvement.

[2023/12/03] added support of **ChatGLM**, **QWen**, **Baichuan**, **Mistral**, **InternLM**!

[2023/12/02] added support for safetensors. Now support all top 10 models in open llm leaderboard.

[2023/12/01] airllm 2.0. Support compressions (block-wise quantization).

[2023/11/20] airllm Initial version!

## Star History

<a href="https://star-history.com/#lyogavin/airllm&Timeline">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/star-history-dark.png">
    <img alt="Star History Chart" src="assets/star-history.png">
  </picture>
</a>

## Table of Contents

* [Quick start](#quickstart)
* [Model Compression](#model-compression)
* [Configurations](#configurations)
* [Run on MacOS](#macos)
* [Example notebooks](#example-python-notebook)
* [Supported Models](#supported-models)
* [Training](#training)
* [Acknowledgement](#acknowledgement)
* [FAQ](#faq)

## Quickstart

### 1. Install package

First, install the airllm pip package.

```bash
pip install airllm
```

### 2. Inference

Then, initialize AirLLMLlama2, pass in the huggingface repo ID of the model being used, or the local path, and inference can be performed similar to a regular transformer model.

(*You can also specify the path to save the split layered model through **layer_shards_saving_path** when init AirLLMLlama2.*

```python
from airllm import AutoModel

MAX_LENGTH = 128
# just pass a hugging face repo id — works with almost any popular model:
model = AutoModel.from_pretrained("Qwen/Qwen3-32B")

# go bigger with the exact same one line:
#model = AutoModel.from_pretrained("Qwen/Qwen3.8-27B")          # 27B dense VL, 3.33GB
#model = AutoModel.from_pretrained("Qwen/Qwen3.8-Flash-Next")    # 125B MoE + 51B PLE, 5.95GB
#model = AutoModel.from_pretrained("Qwen/Qwen3-235B-A22B")     # 235B, runs in ~3GB
#model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V3")  # 671B, runs in ~12GB

# or use a model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/...")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].to(model.device), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```
 
 
Note: During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the huggingface cache directory.
 

## Model Compression

Layers are stored on disk with block-wise quantization, so each token reads 2x (8-bit) or 4x
(4-bit) fewer bytes. The speedup is however large the disk read is of your per-token time — if you
are already compute-bound or your layers are being served from the cache, compression buys you
less, and it costs a little accuracy. (Background on why block-wise quantization holds up:
[this paper](https://arxiv.org/abs/2212.09720).)

![speed_improvement](https://github.com/lyogavin/airllm/blob/main/assets/airllm2_time_improvement.png?v=2&raw=true)

#### How to enable model compression speed up:

* Step 1. make sure you have [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) installed by `pip install -U bitsandbytes `
* Step 2. make sure airllm version later than 2.0.0: `pip install -U airllm` 
* Step 3. when initialize the model, passing the argument compression ('4bit' or '8bit'):

```python
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct",
                     compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
                    )
```

#### Compression and layer caching

These two settings pull in opposite directions. Compression makes each layer cheaper to *read*;
caching makes layers cost *nothing* to read by keeping them in memory. They compose, and the
engine picks the better of the two per layer rather than forcing a choice:

* A layer already in the cache is never read from disk, so its compression format is irrelevant.
* A cached layer is kept in its expanded form, so 4-bit does not shrink what the cache holds —
  budget by the expanded size, not the file size.
* If everything fits in the cache, compression only costs accuracy. If nothing fits, compression
  is the thing that helps.

For a model that fits in VRAM plus pinned host memory, run with the cache on and compression off.
For a model far larger than memory, turn compression on.

### Shrinking the checkpoint on disk

`compression=` above trades accuracy for a smaller shard. The `codec=` argument is the wider tool:
it can also make a checkpoint **smaller without changing a single output bit**.

```python
# Lossless: the model behaves exactly as it would uncompressed. ~1.5x smaller on bf16.
model = AutoModel.from_pretrained("Qwen/Qwen3-32B", codec="lossless")

# Explicitly lossy, when you need a bigger reduction and accept the drift.
model = AutoModel.from_pretrained("Qwen/Qwen3-32B", codec="fp8")       # 2x
model = AutoModel.from_pretrained("Qwen/Qwen3-32B", codec="4bit")      # 3.5x

# "Make it fit in N GB": pick the least-lossy codec that meets the budget.
model = AutoModel.from_pretrained("Qwen/Qwen3-32B", codec="auto", size_budget_gb=12)
```

| codec | ratio (bf16) | fidelity |
| --- | --- | --- |
| `none` | 1.00x | bit-exact |
| `exponent` / `lossless` | ~1.25x | **bit-exact** |
| `nvcomp` | ~1.25x | **bit-exact** (accepts NVIDIA nvCOMP when installed) |
| `fp8` | 2.0x | ~2% relative error |
| `8bit` | 2.0x | ~1% relative error |
| `4bit` | 3.5x | ~9% relative error |
| `ternary` | 7.5x | ~38% relative error (crude; see below) |

The lossless codecs work on the same observation NVIDIA's nvCOMP and DFloat11 exploit: a bfloat16
exponent holds about 2.6 bits of real information in its 8 bits, and a weight matrix uses only ~20-31
distinct exponents. AirLLM stores the exponent as a 5-bit index into a per-layer alphabet and keeps
sign+mantissa verbatim, so reconstruction is exact — including infinities, NaN and denormals, which
are covered by tests. Because it is exact, it composes with everything else in the engine: the
streaming hooks, prefetching, and the layer cache all see ordinary tensors.

One deliberate trade: entropy coding (Huffman/ANS) reaches ~1.5x instead of ~1.25x, but it is
inherently serial — a symbol's position depends on every previous length — and a decoder layer has
~10⁸ symbols, so it took minutes per layer and made a model load look hung. The fixed-width alphabet
costs about 2 bits per weight more and runs as a single vectorised pass: ~13s to encode and ~15s to
decode a 0.17GB layer, with peak memory bounded regardless of layer size.

Guarantees the planner enforces:

* **Compression is never silently lossy.** `codec="auto"` resolves to the strongest lossless codec.
  It only steps down to `fp8`/`4bit`/`ternary` when a `size_budget_gb` cannot otherwise be met, and
  `allow_lossy=False` forbids even that.
* **Norms are never quantized.** They are the smallest tensors in the model and they scale every
  downstream activation.
* **A codec that would grow a shard stores it verbatim instead.** Sub-gigabyte models and
  already-compact integer payloads (packed 4-bit weights, fp8 block scales) pass through untouched.
* Each codec writes its own `splitted_model.codec-<name>` directory, so switching codecs never
  silently reuses an incompatible split.

#### Why not ternary by default

Bonsai 2-class compression gets a 27B model to ~1.7 bits per weight, and the tempting reading is
"apply that to any checkpoint." The measured result says otherwise: post-hoc ternary on a trained
weight matrix reconstructs to ~38% relative error, because ~40% of the weights fall on a grid a
three-value code cannot represent at all. The published results come from *quantization-aware
training* — the network adapts to the grid — which is not something that can be recovered from a
checkpoint. An orthogonal Hadamard rotation (the other half of that recipe) is implemented and
helps *scalar* codecs measurably — it collapses row-scale spread by ~19x — but it does not rescue
ternary either. So `ternary` is available, labelled, and never chosen automatically.

#### What are the differences between model compression and quantization?

Quantization normally needs to quantize both weights and activations to really speed things up. Which makes it harder to maintain accuracy and avoid the impact of outliers in all kinds of inputs.

While in our case the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller. So, we get to only quantize the weights' part, which is easier to ensure the accuracy.

## Performance

Three changes matter more than any kernel-level tuning, because they remove work rather than speed
it up:

1. **Don't re-read what is still in memory.** See [Layer caching](#layer-caching).
2. **Actually overlap the disk read.** The reader thread was being handed each layer only after
   the previous one had been consumed, and a cancelled-but-already-started read was re-issued on
   the main thread, so every layer cost two reads. With both fixed, a read slower than a layer's
   compute now hides behind it: on a simulated 30 ms/layer disk, 3 tokens took 1432 ms with
   prefetching off and 763 ms with it on.
3. **Don't run a full garbage collection per layer.** The streaming hooks used to call
   `gc.collect()` + `malloc_trim()` + `cuda.empty_cache()` after *every* module, ~80 times per
   generated token for a 70B model. On a large Python heap that costs tens of milliseconds on its
   own. Eviction is now a plain release, and the allocator is only flushed when memory is actually
   short.

Measured on a CPU-only sandbox with a 12-layer toy Llama, 16 generated tokens:

| | time per token |
| --- | --- |
| before | 1433 ms |
| after | 10 ms |

The same mechanism dominates on GPU: the removed `gc.collect()` is CPU work that runs while the
GPU sits idle, so it appears directly in time-per-token. Caching then removes the disk reads.

Compression deserves a note, because it is the one setting that is now usually the *wrong* answer.
It was introduced to cut the bytes each token reads; the cache cuts those bytes to zero. Prefer the
cache, and reach for compression when the model is too large for the host tiers as well.

## Configurations
 
When initialize the model, we support the following configurations:

* **compression**: supported options: 4bit, 8bit for 4-bit or 8-bit block-wise quantization, or by default None for no compression. Works together with prefetching and layer caching, and runs on CPU as well as CUDA.
* **profiling_mode**: supported options: True to output time consumptions or by default False
* **layer_shards_saving_path**: optionally another path to save the split model
* **hf_token**: huggingface token can be provided here if downloading gated models like: *meta-llama/Llama-2-7b-hf*
* **prefetching**: prefetching to overlap the model loading and compute. By default, turned on.
* **prefetch_depth**: how many layers to read ahead (default 2). One leaves the reader no lookahead, so every read is waited on rather than overlapped; raise it further when a layer takes much longer to read than to compute. Each queued read holds one layer in host memory.
* **layer_cache_gb**: VRAM budget, in GB, for keeping recently used layers loaded. Default: auto (a fraction of free VRAM). See [Layer caching](#layer-caching).
* **park_cache_gb**: pinned-host budget, in GB, for keeping an evicted layer's host copy so it can be restored with a memcpy instead of a disk read. Default: auto.
* **full_load**: load the whole model instead of streaming it. Default: auto — enabled only when the checkpoint comfortably fits.
* **cache_plan_path**: JSON file recording which layers were cached, so a warm start resumes the same set instead of re-deriving it.
* **progress**: print stage transitions and throughput to stderr, so a long run is distinguishable from a hang. Default: False.
* **stall_timeout**: raise if a stage makes no progress for this many seconds (useful on network filesystems).
* **delete_original**: if you don't have too much disk space, you can set delete_original to true to delete the original downloaded hugging face model, only keep the transformed one to save half of the disk space. 

## Layer caching

Streaming reads the checkpoint from disk on every pass. Generation makes one pass *per token*, so
a 70B model re-reads ~140GB for every token it emits — the single biggest cost in AirLLM, far
larger than the arithmetic. The engine therefore caches streamed layers in two tiers:

| Tier | Holds | A cache hit costs | Pays off when |
| --- | --- | --- | --- |
| **VRAM** (`layer_cache_gb`) | fully loaded layers | nothing | the GPU can spare the memory |
| **Pinned host** (`park_cache_gb`) | evicted layers' host copies | one host→device memcpy | always (costs no VRAM) |

Both budgets are byte-based and derived from what is actually free at start-up, so caching can
never be the reason a model fails to fit: a layer larger than the budget simply is not cached.
The cache is an LRU, biased towards the head of the stack — when the budget cannot hold every
layer, caching the *start* means the next token begins computing immediately instead of
immediately waiting on disk.

```python
# default: sized automatically
model = AutoModel.from_pretrained("Qwen/Qwen3-32B")

# a 24GB card running a 13B model: keep as much as fits, up to 10GB
model = AutoModel.from_pretrained("Qwen/Qwen3-32B", layer_cache_gb=10)

# a 4GB card: no room in VRAM, but pinned host parking still removes the disk read
model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", layer_cache_gb=0)

# measure the uncached path
model = AutoModel.from_pretrained("...", layer_cache_gb=0, park_cache_gb=0)
```

Three related behaviours:

* `full_load="auto"` (the default) loads the whole checkpoint when it comfortably fits, and
  generation runs at plain-`transformers` speed. Streaming is only used when it has to be.
* `model.layer_cache_stats()` reports hits, misses and peak VRAM; `model.clear_layer_cache()`
  releases the cached memory.
* `model.save_cache_plan(path)`, then `cache_plan_path=path` on the next start, primes the
  read-ahead with the layers that proved worth keeping last time. A plan is a hint, not a saved
  copy of the weights: the layers are still read, but all of them are read up front and overlapped
  with compute, so one pass leaves the cache warm instead of re-deriving the working set layer by
  layer.

With `progress=True` every stage prints to stderr:

```
[airllm] generate: max_new_tokens=20
[airllm] generate finished: 76 cache hits / 4 loads (95% hit rate), 0.31 tok/s
```

## Supported models

Any `*ForCausalLM` that `transformers` can load, streamed without a per-architecture subclass. The
streaming layout — which modules to load, in what order — is read from the instantiated model's
module tree rather than hard-coded per family, so a model whose trunk is named `transformer.*`,
`gpt_neox.*` or `model.decoder.*` works exactly like one named `model.*`.

Verified token-for-token against `transformers`, with and without a lossless codec:

| decoder container | families |
| --- | --- |
| `model.layers` | Llama, Qwen2/Qwen3, Mistral, Mixtral, Qwen2-MoE, Gemma, Phi, StableLM, Cohere, OLMo |
| `transformer.h` | GPT-2, BLOOM, Falcon, GPT-J, GPT-BigCode / StarCoder |
| `gpt_neox.layers` | GPT-NeoX / Pythia |
| `model.decoder.layers` | OPT |

Discovery handles the cases that break a fixed layout, each of which has its own regression test:

* **No trunk-level final norm** (GPT-2, Falcon): the norm stage becomes a no-op slot instead of
  asking for a shard that does not exist.
* **A norm module with no parameters** (OLMo): the module is real but has no weights, so no shard is
  written for it. Detected by inspecting the module, not by a per-family flag.
* **Learned position tables** (GPT-2 `wpe`, OPT `embed_positions`): must stay resident, never
  streamed.
* **A stage before the first block** (BLOOM's post-embedding norm): gets its own shard so it is not
  stranded on the meta device.

`AutoModel.describe_support()` prints the same summary, and `AutoModel.supported_architectures()`
lists the families with hand-written handling for something the generic path cannot infer.

## Dense and sparse (MoE)

Dense and Mixture-of-Experts models both stream, and both are verified token-for-token. What
differs is how much of a sparse layer has to be resident, and that depends on how the checkpoint
stores its experts rather than on the model family:

* **One module per expert** — the experts live in a container (`block_sparse_moe.experts.0`, …) and
  the router calls exactly the ones a token picks. A hook on each expert loads only the routed
  experts' shards, so resident memory tracks the number of *active* experts, not the total. This is
  what makes Kimi K3's 2.8T parameters fit in 3.72GB.
* **One batched tensor for all experts** — current `transformers` packs the experts into a single
  module holding 3-D tensors (`MixtralExperts.gate_up_proj` is `(n_experts, 2*intermediate,
  hidden)`), and the routing index is consumed inside one op. No hook can tell which slice a token
  will read, so the whole layer streams. The model is still *computed* sparsely — only the routed
  experts' slice runs — but the layer is *read* densely, and resident memory is one full layer
  rather than one expert.

AirLLM detects which layout a model has and prints the answer when it starts, because the
difference is invisible in the output and otherwise looks like a size budget that isn't working.
Per-expert streaming is not enabled for the packed layout, and the model's forward is deliberately
not patched to force it: the saving would be a handful of experts out of four.

## MacOS

Just install airllm and run the code the same as on linux. See more in [Quick Start](#quickstart).

* make sure you installed [mlx](https://github.com/ml-explore/mlx?tab=readme-ov-file#installation) and torch
* you probably need to install python native see more [here](https://stackoverflow.com/a/65432861/21230266)
* only [Apple silicon](https://support.apple.com/en-us/HT211814) is supported
* only Llama-style models (`LlamaForCausalLM`, `MistralForCausalLM`) run on macOS for now; other architectures raise an error before anything is downloaded

Example [python notebook] (https://github.com/lyogavin/airllm/blob/main/air_llm/examples/run_on_macos.ipynb)


## Example Python Notebook

Example colabs here:

<a target="_blank" href="https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_all_types_of_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### example of other models (ChatGLM, QWen, Baichuan, Mistral, etc):

<details>


* ChatGLM:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
generation_output = model.generate(
    input_tokens['input_ids'].to(model.device), 
    max_new_tokens=5,
    use_cache= True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```

* QWen:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].to(model.device), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


* Baichuan, InternLM, Mistral, etc:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
#model = AutoModel.from_pretrained("internlm/internlm-20b")
#model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].to(model.device), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


</details>


#### To request other model support: [here](https://docs.google.com/forms/d/e/1FAIpQLSe0Io9ANMT964Zi-OQOq1TJmnvP-G3_ZgQDhP7SatN0IEdbOg/viewform?usp=sf_link)



## Supported Models

AirLLM works out of the box with **virtually every popular open LLM** — just pass its Hugging Face ID to `AutoModel.from_pretrained(...)`. That covers all the major families:

**Llama** (2 / 3 / 3.1 / 3.3 / 4) · **Qwen** (1 / 2 / 2.5 / 3 / 3.5 / 3.8, including MoE, Flash-Next, FP8, and native VL) · **DeepSeek** (V2 / V3 / R1) · **Mistral & Mixtral** · **Phi** · **Gemma** · **ChatGLM** · **Baichuan** · **InternLM** · **Yi** · **Kimi K3** — and most new models the day they're released.

### Tiny GPU, huge models

The trick: AirLLM only ever keeps **one layer on the GPU at a time**, so the VRAM you need depends on the model's layer size — not its total size. That's how a 671B model fits on a hobbyist card:

| Model | Size | GPU VRAM |
|---|---|---|
| Qwen3 / Mistral / Phi (≈8B) | 8B | **~1–2 GB** |
| Qwen3-30B / Mixtral (MoE) | 30–47B | **~1–3 GB** |
| Qwen3.8-27B (dense VL) | 27B | **3.33 GB** |
| Qwen3.8-Flash-Next (MoE + PLE) | ~180B | **5.95 GB** |
| Qwen3-235B (MoE) | 235B | **~3 GB** |
| Llama 3.x 70B (full precision) | 70B | **~4 GB** |
| Llama 3.1 405B | 405B | **~8 GB** |
| DeepSeek-V3 | **671B** | **~12 GB** |

Same one line of code for all of them — no special setup.

## Training

AirLLM can fine-tune huge models on a small GPU. Frozen base weights stream from disk one decoder layer at a time; only the adapters stay resident. **Qwen3.8-Flash-Next (125B)** trains under **6GB**; **Qwen3.8-27B** trains in **~2GB** at seq 512.

This is not Hugging Face Trainer / bitsandbytes QLoRA. Flash-Next needs a `transformers` build with in-tree `qwen4_exp` (`pip install git+https://github.com/huggingface/transformers.git` today).

### 1. Prepare a dataset

One JSON object per line (`.jsonl`). The usual field is `text` — next-token prediction over the whole string:

```json
{"text": "Your first training document. Can be a few sentences or a few paragraphs."}
{"text": "Your second training document."}
```

Instruction pairs work too. Loss is applied on the completion only:

```json
{"prompt": "What is AirLLM?", "completion": "A library that runs and trains huge models on small VRAM."}
{"instruction": "Translate to English", "input": "bonjour", "output": "hello"}
```

A `.txt` file is also fine: one example per blank-line-separated block. A two-line starter file lives at `air_llm/examples/sft_example.jsonl`.

### 2. Run training

From the repo root, point `--data` at your file:

```bash
python air_llm/examples/train_qwen38_flash_next_lora.py \
  --data my_data.jsonl \
  --seq-len 512 \
  --epochs 1 \
  --save-adapter qwen38-flash-next-lora.pt
```

For the 27B dense model:

```bash
python air_llm/examples/train_qwen38_lora.py \
  --data my_data.jsonl \
  --seq-len 512 \
  --epochs 1 \
  --save-adapter qwen38-27b-lora.pt
```

`--steps N` stops after N examples (useful for a smoke test). Omit `--data` and the script overfits a built-in snippet.

### Python API

```python
from airllm import AirLLMLoRAQwen4Exp

trainer = AirLLMLoRAQwen4Exp(
    "Qwen/Qwen3.8-Flash-Next",
    max_seq_len=512,
    lora_r=16,
    delete_original=True,
)

tok = trainer.tokenizer
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

encoded = tok(
    "Your training text here.",
    return_tensors="pt",
    truncation=True,
    max_length=512,
)
loss = trainer.train_step(
    encoded["input_ids"].to(model.device),
    attention_mask=encoded.get("attention_mask"),
)
print(loss)
trainer.save_adapter("qwen38-flash-next-lora.pt")
```

`AirLLMLoRA` is the same API for `Qwen/Qwen3.8-27B`.

## Acknowledgement

A lot of the code are based on SimJeg's great work in the Kaggle exam competition. Big shoutout to SimJeg:

[GitHub account @SimJeg](https://github.com/SimJeg), 
[the code on Kaggle](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), 
[the associated discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414).


## FAQ

### 1. MetadataIncompleteBuffer

safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer

If you run into this error, most possible cause is you run out of disk space. The process of splitting model is very disk-consuming. See [this](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12). You may need to extend your disk space, clear huggingface [.cache](https://huggingface.co/docs/datasets/cache) and rerun. 

### 2. ValueError: max() arg is an empty sequence

Most likely you are loading QWen or ChatGLM model with Llama2 class. Try the following:

For QWen model: 

```python
from airllm import AutoModel #<----- instead of AirLLMLlama2
AutoModel.from_pretrained(...)
```

For ChatGLM model: 

```python
from airllm import AutoModel #<----- instead of AirLLMLlama2
AutoModel.from_pretrained(...)
```

### 3. 401 Client Error....Repo model ... is gated.

Some models are gated models, needs huggingface api token. You can provide hf_token:

```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", #hf_token='HF_API_TOKEN')
```

### 4. ValueError: Asking to pad but the tokenizer does not have a padding token.

Some model's tokenizer doesn't have padding token, so you can set a padding token or simply turn the padding config off:

 ```python
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False  #<-----------   turn off padding 
)
```

## Citing AirLLM

If you find
AirLLM useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{airllm2023,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/airllm/},
  version = {0.0},
  year = {2023},
}
```


## Contribution 

Welcomed contributions, ideas and discussions!

If you find it useful, please ⭐ or buy me a coffee! 🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
