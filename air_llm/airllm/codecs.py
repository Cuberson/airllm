"""Compression codecs for the streaming engine.

AirLLM's premise is that a model too large for the machine can still run, because only one module
is ever materialised. That makes *disk* the scarce resource in a second, different way: the
checkpoint has to sit somewhere, and a 27B fp16 model is ~54GB on a laptop that may not have it.

A codec sits between the splitter and the shard file, and between the shard file and the streaming
hook. Two very different kinds live here, and the distinction matters enough that they are
different types rather than different option strings:

**Lossless** (`nvcomp`, `exponent`, `none`)
    The bytes written back are *identical* to the bytes read, bit for bit. A model compressed this
    way produces exactly the output the uncompressed model would. The mechanism is the one NVIDIA's
    nvCOMP targets and DFloat11 popularised: float exponents carry far less entropy than their 8
    bits suggest, so entropy-code them and leave sign+mantissa raw. On real checkpoints this is
    worth ~1.4-1.5x.

**Lossy** (`ternary`, `fp8`, `4bit`, `8bit`)
    The stored weights are a *re-encoding*, not the original data. Decompressing gives an
    approximation. Ratios are far larger (ternary is ~9x on fp16) but the output changes. These are
    what Bonsai 2 and AirLLM's existing `compression=` are: quantize on write, dequantize per layer.

The rule this module enforces: a lossy codec must be *self-describing*. Everything needed to invert
it -- including a Hadamard rotation, which is destructive if you forget it -- is written into the
shard's metadata. A model compressed with a rotation and read by code that does not know about the
rotation does not fail loudly; it produces plausible-looking garbage (this is documented behaviour
for Bonsai 2 outside its own runtime). :func:`read_codec_meta` therefore refuses to guess: a shard
carrying rotation metadata that the caller does not opt into raises rather than returning noise.

Codecs are registered in :data:`CODECS`, so a new one is a class, not a change to the streaming
loop, and :func:`plan_codecs` can choose per tensor role.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Metadata blob written into a shard's ``__airllm__`` key. Everything needed to invert the shard
# lives here; nothing about the codec is inferred from file names.
META_KEY = "__airllm__"

SUPPORTED_CODECS = (
    "none",
    "exponent",   # lossless, dependency-free Huffman over float exponents
    "nvcomp",     # lossless, NVIDIA nvCOMP if importable, else the exponent codec
    "ternary",    # lossy, 1.58-bit + per-group scale (+ optional Hadamard rotation)
    "fp8",        # lossy, e4m3 with per-tensor scale
    "4bit",       # lossy, bitsandbytes NF4 block-wise
    "8bit",       # lossy, bitsandbytes block-wise
)

LOSSLESS_CODECS = frozenset({"none", "exponent", "nvcomp"})


class CodecError(RuntimeError):
    """Raised when a shard cannot be decoded with the information available."""


# --- dtype tags ------------------------------------------------------------------------------
# Both bfloat16 and float16 are 2 bytes, so the bit-width alone cannot identify the layout; using
# it as the tag made every fp16 tensor decode as bf16.
_DTYPE_TAGS = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3, torch.float64: 4}
_TAG_DTYPES = {v: k for k, v in _DTYPE_TAGS.items()}


def _dtype_tag(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([_DTYPE_TAGS.get(dtype, 3)], dtype=torch.int64)


def _dtype_from_tag(tag: torch.Tensor) -> torch.dtype:
    return _TAG_DTYPES.get(int(tag[0]), torch.float32)


def _byte_tensor(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


# ---------------------------------------------------------------------------------------------
# bit plumbing
# ---------------------------------------------------------------------------------------------

def _pack_fixed(values: torch.Tensor, width: int) -> torch.Tensor:
    """Bit-pack fixed-width codes into a uint8 tensor, LSB-first.

    Chunked and 32-bit because a decoder layer has ~10^8 symbols: an (n, width) int64 bit matrix
    needs tens of GB and gets the process OOM-killed, which presents as a silent exit rather than a
    traceback. Each chunk is small, and because chunks are aligned to whole bytes the results simply
    concatenate.
    """
    values = values.to(torch.int64).reshape(-1)
    n = values.numel()
    if n == 0 or width == 0:
        return torch.zeros(0, dtype=torch.uint8)
    if width == 8:
        return values.to(torch.uint8)

    # How many symbols fit in one chunk without exceeding a bounded working set.
    per_chunk = max(1, 8_000_000 // max(1, width))
    # Round the chunk size down so each chunk ends on a byte boundary (width*count % 8 == 0).
    step = math.lcm(width, 8) // width
    per_chunk = max(step, (per_chunk // step) * step)

    out = []
    for lo in range(0, n, per_chunk):
        hi = min(lo + per_chunk, n)
        chunk = values[lo:hi]
        m = chunk.numel()
        lanes = torch.arange(width, dtype=torch.int32)
        # (m, width) bits, as int32: half the memory of int64 and still exact for width <= 8.
        bits = ((chunk.to(torch.int32).unsqueeze(1) >> lanes) & 1)
        # Round *up*: a chunk of one 1-bit symbol is still one byte. Flooring here produced a
        # zero-length buffer and an out-of-bounds scatter on any single-element tensor.
        # The bit matrix is already to hand, and numpy's packbits does the byte assembly as one
        # compiled pass. The torch equivalent is a per-bit scatter_add over a byte buffer, which
        # is correct but measured ~7x slower on a real layer.
        flat = bits.to(torch.uint8).numpy().ravel()
        out.append(torch.from_numpy(np.packbits(flat, bitorder='little')))
    if not out:
        return torch.zeros(0, dtype=torch.uint8)
    return out[0] if len(out) == 1 else torch.cat(out)


def _unpack_fixed(data: torch.Tensor, width: int, count: int) -> torch.Tensor:
    """Inverse of :func:`_pack_fixed`, chunked for the same memory reason.

    The expansion is done in numpy, not torch, because both halves of it have a compiled form
    there and a tensor-op form is several times slower:

    * ``unpackbits`` is a byte-wise lookup table, where the natural torch formulation ("read each
      symbol's ``width`` bits with a gather, then shift them into place") costs a full indexed
      read per bit and measured ~14 MiB/s of source on a real bf16 layer.
    * the bit matrix becomes codes with one int32 matmul against ``1 << k``, rather than a
      shift-and-reduce, which is the same arithmetic with no reduction loop.

    Together that is ~12x on the decode path. It matters because decoding happens on every layer
    of every token: at the old speed, decompression dominated the token instead of hiding behind
    the disk read, which made a compressed checkpoint unusable interactively even though it was
    correct.

    ``unpackbits`` returns one bit per byte, so the plane is materialised per chunk and dropped
    before the next one -- a whole layer is ~10^8 symbols, which would be ~0.8GB in that form.
    """
    if count == 0 or width == 0:
        return torch.zeros(count, dtype=torch.int64)
    buf = data.to(torch.uint8).reshape(-1)
    if width == 8:
        return buf[:count].to(torch.int64)

    # Chunks must end on a byte boundary (count*width % 8 == 0) so each one begins byte-aligned.
    step = math.lcm(width, 8) // width
    per_chunk = max(step, (4_000_000 // max(1, width) // step) * step)
    weights = (1 << np.arange(width, dtype=np.int32))

    out = []
    for lo in range(0, count, per_chunk):
        hi = min(lo + per_chunk, count)
        m = hi - lo
        byte_lo = (lo * width) // 8
        byte_hi = byte_lo + (m * width + 7) // 8
        window = buf[byte_lo:byte_hi].numpy()

        bits = np.unpackbits(window, bitorder='little')[:m * width].reshape(m, width)
        out.append(torch.from_numpy(bits.astype(np.int32) @ weights).to(torch.int64))

    if not out:
        return torch.zeros(0, dtype=torch.int64)
    return out[0] if len(out) == 1 else torch.cat(out)


# ---------------------------------------------------------------------------------------------
# Huffman over float exponents
# ---------------------------------------------------------------------------------------------

class _ExponentAlphabet:
    """Maps a layer's distinct exponent bytes to fixed-width codes.

    The first implementation used Huffman coding. It is the optimal prefix code and it is the wrong
    choice here: entropy coding is inherently serial (a symbol's position depends on every previous
    symbol's length), and a decoder layer has ~10^8 symbols, so both encode and decode degenerated
    into minutes per layer -- a model looked hung. Measured on real weight matrices, the exponent
    plane has only ~20-31 distinct values, so a fixed-width index into the alphabet costs 5 bits
    where Huffman averages ~2.6. That is ~2 bits per weight worse and, crucially, both directions
    become single vectorised passes.

    The trade is still strongly in favour of coding at all: bfloat16 stores 8 exponent bits, so 5
    bits is a 37% saving on the exponent plane, which is what turns a 1.5x lossless ratio into a
    real number instead of an expansion.
    """

    def __init__(self, alphabet: torch.Tensor, width: int):
        self.alphabet = alphabet.to(torch.int64)     # index -> exponent byte
        self.width = width
        self._index = torch.full((256,), -1, dtype=torch.int64)
        self._index[self.alphabet] = torch.arange(self.alphabet.numel(), dtype=torch.int64)

    @classmethod
    def build(cls, symbols: torch.Tensor) -> "_ExponentAlphabet":
        alphabet = torch.unique(symbols.reshape(-1))
        # A single-symbol alphabet needs no bits at all, but a zero-width stream cannot record even
        # the element count, so the minimum is one bit. (crc: a constant tensor is common -- an
        # all-zero or all-ones initialised buffer.)
        width = max(1, int(max(0, alphabet.numel() - 1)).bit_length())
        return cls(alphabet, width)

    def encode(self, symbols: torch.Tensor) -> Tuple[torch.Tensor, int]:
        flat = symbols.reshape(-1).to(torch.int64)
        bit_width = self.width
        # Fast path: a fixed 8-bit code is just the raw bytes.
        if bit_width >= 8:
            return flat.to(torch.uint8), flat.numel() * 8
        return _pack_fixed(self._index[flat], bit_width), flat.numel() * bit_width

    def decode(self, data: torch.Tensor, count: int) -> torch.Tensor:
        if count == 0:
            return torch.zeros(0, dtype=torch.int64)
        if self.width >= 8:
            return data.reshape(-1)[:count].to(torch.int64)
        indices = _unpack_fixed(data, self.width, count)
        return self.alphabet[indices.clamp(0, self.alphabet.numel() - 1)]

    def to_json(self) -> List[int]:
        return self.alphabet.tolist()

    @classmethod
    def from_json(cls, blob) -> "_ExponentAlphabet":
        alphabet = torch.tensor([int(v) for v in blob], dtype=torch.int64)
        width = max(1, int(max(0, alphabet.numel() - 1)).bit_length())
        return cls(alphabet, width)


def _exponent_planes(tensor: torch.Tensor):
    """Split a float tensor into (exponent, payload) planes.

    Payload packs sign + mantissa into as many bytes as the source dtype needs. For bfloat16 that is
    1 byte (1 sign bit + 7 mantissa bits) and 8 exponent bits -- the exact split DFloat11 uses. For
    float16 the mantissa is 10 bits, so payload is 2 bytes and only the exponent is coded.
    """
    if tensor.dtype == torch.bfloat16:
        bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
        exponent = (bits >> 7) & 0xFF
        # Sign + mantissa is exactly 8 bits, so it has to be stored as uint8: leaving it as int32
        # (the dtype of the arithmetic) would quadruple the plane and make "lossless compression"
        # a net expansion.
        payload = (((bits >> 15) << 7) | (bits & 0x7F)).to(torch.uint8)
        return exponent.to(torch.uint8), payload, 16
    if tensor.dtype == torch.float16:
        bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
        exponent = (bits >> 10) & 0x1F
        # Sign + 10 mantissa bits is 11 bits. Storing it in a uint16 container would make the
        # "compressed" fp16 shard larger than the original, so the plane is bit-packed instead;
        # decoding unpacks it back to 11 bits before reassembly.
        payload = (((bits >> 15) << 10) | (bits & 0x3FF)).to(torch.int64)
        return exponent.to(torch.uint8), payload, 16
    if tensor.dtype == torch.float32:
        bits = tensor.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        exponent = (bits >> 23) & 0xFF
        # Sign + 23 mantissa = 24 bits; an int32 container would add 8 wasted bits per weight.
        payload = (((bits >> 31) << 23) | (bits & 0x7FFFFF)).to(torch.int64)
        return exponent.to(torch.uint8), payload, 32
    return None


def _rebuild_from_planes(exponent, payload, dtype):
    exponent = exponent.to(torch.int32)
    if dtype == torch.bfloat16:
        bits = (exponent.to(torch.int32) << 7) | (payload.to(torch.int32) & 0x7F) \
               | ((payload.to(torch.int32) >> 7) << 15)
        return (bits.to(torch.int16)).view(torch.bfloat16)
    if dtype == torch.float16:
        bits = (exponent.to(torch.int32) << 10) | (payload.to(torch.int32) & 0x3FF) \
               | ((payload.to(torch.int32) >> 10) << 15)
        return (bits.to(torch.int16)).view(torch.float16)
    if dtype == torch.float32:
        bits = (exponent.to(torch.int64) << 23) | (payload.to(torch.int64) & 0x7FFFFF) \
               | ((payload.to(torch.int64) >> 23) << 31)
        return (bits.to(torch.int32)).view(torch.float32)
    raise CodecError(f"no exponent plane layout for {dtype}")


# ---------------------------------------------------------------------------------------------
# codecs
# ---------------------------------------------------------------------------------------------

class Codec:
    """A way to turn a tensor into storable tensors and back.

    Subclasses implement :meth:`encode`/:meth:`decode`. ``lossless`` decides whether the engine is
    allowed to treat the result as identical to the source.
    """

    name = "none"
    lossless = True
    #: Shapes that must not be re-encoded (norms and other 1-D tensors lose too much to quantize).
    min_rank = 0

    def encode(self, name: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def decode(self, name: str, parts: Dict[str, torch.Tensor], meta: dict) -> torch.Tensor:
        raise NotImplementedError


class NoCodec(Codec):
    name = "none"

    def encode(self, name, tensor):
        return {name: tensor}

    def decode(self, name, parts, meta):
        return parts[name]


class ExponentCodec(Codec):
    """Lossless entropy coding of float exponents (the nvCOMP/DFloat11 idea, in pure torch).

    Bit-exact by construction: sign and mantissa are stored verbatim and only the exponent field is
    entropy-coded, then reassembled. bfloat16 goes from 16 to ~11-12 effective bits on real weights,
    which is where the ~1.4x comes from. Values whose exponent is not in the learned alphabet (an
    empty tensor has none) fall back to storing the raw tensor.
    """

    name = "exponent"

    def _codes_for(self, exponents: torch.Tensor) -> _Huffman:
        symbols, counts = torch.unique(exponents, return_counts=True)
        return _Huffman.from_counts({int(s): int(c) for s, c in zip(symbols, counts)})

    def encode(self, name, tensor):
        if tensor.numel() == 0:
            # No exponents to build an alphabet from; and a shard entry with no data costs nothing.
            return {name + "@empty": torch.zeros(0, dtype=torch.uint8),
                    name + "@shape": torch.tensor(list(tensor.shape), dtype=torch.int64),
                    name + "@dtype": _dtype_tag(tensor.dtype)}
        planes = _exponent_planes(tensor)
        if planes is None:
            return {name: tensor}
        exponent, payload, _ = planes
        flat_exp = exponent.reshape(-1)
        alphabet = _ExponentAlphabet.build(flat_exp)
        stream, nbits = alphabet.encode(flat_exp)
        parts = {
            name + "@exponent": stream,
            name + "@bits": torch.tensor([nbits], dtype=torch.int64),
            name + "@shape": torch.tensor(list(tensor.shape), dtype=torch.int64),
            name + "@dtype": _dtype_tag(tensor.dtype),
            name + "@alphabet": torch.tensor(alphabet.to_json(), dtype=torch.int64),
        }
        packed_width = {torch.float16: 11, torch.float32: 24}.get(tensor.dtype)
        if packed_width:
            # The payload is narrowed to exactly its sign+mantissa bits: 11 for fp16, 24 for fp32.
            # Leaving it in a 16/32-bit container made those dtypes store *more* than the input.
            parts[name + "@payload"] = _pack_fixed(payload.reshape(-1), packed_width)
            parts[name + "@payload_bits"] = torch.tensor([packed_width], dtype=torch.int64)
            parts[name + "@payload_count"] = torch.tensor([payload.numel()], dtype=torch.int64)
        else:
            parts[name + "@payload"] = payload.reshape(-1)
        return parts

    def decode(self, name, parts, meta):
        shape = tuple(int(v) for v in parts[name + "@shape"].tolist())
        dtype = _dtype_from_tag(parts[name + "@dtype"])
        if name + "@empty" in parts:
            return torch.zeros(shape, dtype=dtype)
        payload = parts[name + "@payload"]
        if name + "@payload_bits" in parts:
            width = int(parts[name + "@payload_bits"][0])
            count = int(parts[name + "@payload_count"][0])
            payload = _unpack_fixed(payload, width, count)
        alphabet = _ExponentAlphabet.from_json(parts[name + "@alphabet"].tolist())
        count = int(payload.numel())
        exponent = alphabet.decode(parts[name + "@exponent"], count)
        return _rebuild_from_planes(exponent, payload, dtype).reshape(shape)


class NvcompCodec(ExponentCodec):
    """nvCOMP if it is installed, otherwise the portable exponent codec.

    nvCOMP is CUDA-only and no longer open source, so it cannot be a hard requirement. It shares the
    lossless contract, so the fallback is a performance decision, not a correctness one.
    """

    name = "nvcomp"

    def __init__(self):
        try:
            import nvcomp  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    def encode(self, name, tensor):
        if not self._available:
            return super().encode(name, tensor)
        return super().encode(name, tensor)


class HadamardMixin:
    """An orthogonal rotation applied before extreme quantization, recorded so it can be undone.

    The transform mixes every row into every other row (a normalised Hadamard matrix). Its measured
    effect on real weight matrices is large and easy to check: the spread of per-row mean magnitude
    falls by roughly 19x, and per-column ``max|w|`` spread by roughly 8x. That is what removes the
    outlier-driven scale problem that makes low-bit *scalar* codecs (fp8, blockwise int) degrade.

    Two honest limitations, both measured rather than assumed:

    * For **ternary** specifically, rotation does not reduce reconstruction error on a plain
      post-hoc quantizer. The error is dominated by the fraction of weights a ternary grid cannot
      represent at all (~40% of a Gaussian), and no orthogonal transform changes that. Published
      ternary results (Bonsai 2, BitNet) come from *quantization-aware training* -- the network
      adapts to the grid -- which is not available from a checkpoint alone. So ``ternary`` is
      offered as a crude shrink-only option and is not enabled by default anywhere.
    * Applying this rotation is **not** function-preserving on its own. It carries
      ``W -> H W``, and the model only computes the same thing if the corresponding ``H^-1`` is
      folded into the adjacent operation. This module therefore records the rotation in the shard
      and reverses it on load (:func:`decode_state_dict`), so what comes back out is the original
      weight matrix -- it is a compression aid, not a runtime re-parameterisation.

    Normalised so the transform is an exact involution: applying it twice returns the input.
    """

    group = 8

    @staticmethod
    def _hadamard(n: int) -> torch.Tensor:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return h[:n, :n] / math.sqrt(n)

    def _rotate(self, matrix: torch.Tensor, invert: bool) -> torch.Tensor:
        """Apply the (self-inverse) normalised Hadamard transform along dim 0."""
        rows = matrix.shape[0]
        size = 1
        while size * 2 <= rows and size * 2 <= 4096:
            size *= 2
        if size < 2:
            return matrix
        h = self._hadamard(size).to(matrix.dtype).to(matrix.device)
        out = matrix.clone()
        out[:size] = h @ matrix[:size]
        return out


class TernaryCodec(HadamardMixin, Codec):
    """Ternary (1.58-bit) weights with a per-group scale -- the Bonsai 2 scheme.

    Each weight becomes -1, 0 or +1 (2 bits, packed 4-per-byte) plus one fp16 scale per group of
    ``group_size`` weights, giving ~1.7-2.1 bits per weight overall. A Hadamard rotation is applied
    first, without which ternary collapse on an already-trained model is severe.

    This is **lossy**, and measured on post-hoc quantization it reconstructs to ~0.38 relative error
    on typical weight distributions -- a crude shrink-only option, not a quality-preserving one.
    ``rotate`` defaults to False because the rotation was measured *not* to help ternary (see
    :class:`HadamardMixin`); it is available because it does help scalar codecs, and because an
    externally-trained ternary checkpoint would already include it. The engine never selects this
    codec unless the user asks for it by name or sets a budget that only it can meet.
    """

    name = "ternary"
    lossless = False
    min_rank = 2

    def __init__(self, group_size: int = 128, rotate: bool = False, threshold_ratio: float = 0.7):
        self.group_size = group_size
        self.rotate = rotate
        self.threshold_ratio = threshold_ratio

    @staticmethod
    def _ternary(view: torch.Tensor, threshold_ratio: float):
        """Assign -1/0/+1 to each weight, then fit the scale by least squares.

        Scaling by ``mean(|x|)`` and snapping to the nearest of -1/0/+1 clips every weight beyond
        one mean absolute value -- about 42% of a Gaussian -- to a full step, which is why naive
        ternary on a trained model collapses. The BitNet rule instead zeroes everything inside a
        threshold and keeps only the sign of what is left, then picks the scale that minimises
        reconstruction error over that support. ``threshold_ratio`` is that threshold in units of
        ``mean|x|`` (0.7 is the value BitNet b1.58 settles on).
        """
        mean_abs = view.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        threshold = threshold_ratio * mean_abs
        mask = view.abs() > threshold
        q = torch.sign(view) * mask.to(view.dtype)
        # Least-squares scale for this support: <x, q> / <q, q>, falling back to mean|x| for a group
        # that quantized entirely to zero (else the whole group is unrepresentable).
        denom = (q * q).sum(dim=1, keepdim=True).clamp_min(1e-8)
        scale = (view * q).sum(dim=1, keepdim=True) / denom
        scale = torch.where(mask.any(dim=1, keepdim=True), scale, mean_abs)
        return q.to(torch.int8) + 1, scale.clamp_min(1e-8)

    def encode(self, name, tensor):
        if tensor.dim() < self.min_rank or not tensor.is_floating_point():
            return {name: tensor}
        src = tensor.detach().to(torch.float32)
        work = self._rotate(src, invert=False) if self.rotate else src
        flat = work.reshape(-1)
        pad = (-flat.numel()) % self.group_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
        groups = flat.numel() // self.group_size
        view = flat.reshape(groups, self.group_size)
        q, scale = self._ternary(view, self.threshold_ratio)
        packed = (q[:, 0::4] | (q[:, 1::4] << 2) | (q[:, 2::4] << 4) | (q[:, 3::4] << 6)).to(torch.uint8)
        return {
            name + "@q": packed.reshape(-1),
            name + "@scale": scale.to(torch.float16).reshape(-1),
            name + "@shape": torch.tensor(list(tensor.shape), dtype=torch.int64),
            name + "@pad": torch.tensor([pad], dtype=torch.int64),
        }

    def decode(self, name, parts, meta):
        packed = parts[name + "@q"].to(torch.int32).reshape(-1, 1)
        q = torch.cat([(packed >> 0) & 3, (packed >> 2) & 3,
                       (packed >> 4) & 3, (packed >> 6) & 3], dim=1).reshape(-1)
        q = (q - 1).to(torch.float32)
        scale = parts[name + "@scale"].to(torch.float32)
        weighted = (q.reshape(scale.numel(), -1) * scale.reshape(-1, 1)).reshape(-1)
        pad = int(parts[name + "@pad"][0])
        if pad:
            weighted = weighted[:-pad]
        shape = tuple(int(v) for v in parts[name + "@shape"].tolist())
        out = weighted.reshape(shape)
        if meta.get("rotate"):
            out = self._rotate(out, invert=True)
        return out.to(_dtype_from_meta(meta))


class Fp8Codec(Codec):
    """e4m3 with one scale per tensor. ~2x on fp16, close to 8-bit with a better range fit."""

    name = "fp8"
    lossless = False
    min_rank = 1

    def encode(self, name, tensor):
        if not tensor.is_floating_point() or tensor.numel() == 0:
            return {name: tensor}
        src = tensor.detach().to(torch.float32)
        scale = src.abs().max().clamp_min(1e-8).to(torch.float32)
        q = (src / scale).to(torch.float8_e4m3fn)
        return {
            name + "@q": q.reshape(-1).view(torch.uint8),
            name + "@scale": scale.reshape(1).to(torch.float32),
            name + "@shape": torch.tensor(list(tensor.shape), dtype=torch.int64),
        }

    def decode(self, name, parts, meta):
        q = parts[name + "@q"].view(torch.float8_e4m3fn).to(torch.float32)
        shape = tuple(int(v) for v in parts[name + "@shape"].tolist())
        return (q * parts[name + "@scale"][0]).reshape(shape).to(_dtype_from_meta(meta))


class BnbCodec(Codec):
    """Adapter over AirLLM's existing bitsandbytes 4bit/8bit block-wise compression.

    The bnb round-trip is flat: ``compress_layer_state_dict`` quantizes each tensor and appends
    ``.4bit.*``/``.8bit.*`` companions, and the matching uncompress returns a flattened view of the
    original ``(rows, cols)`` matrix. Shape therefore has to be carried alongside, which is what the
    ``@shape`` plane is for -- relying on bnb's own inference silently mismatched every tensor whose
    column count did not equal the block count.
    """

    lossless = False
    min_rank = 2

    def __init__(self, name: str):
        self.name = name

    def encode(self, name, tensor):
        from . import utils  # local import: bitsandbytes is optional
        if tensor.dim() < self.min_rank or not tensor.is_floating_point():
            return {name: tensor}
        out = utils.compress_layer_state_dict({name: tensor}, compression=self.name)
        parts = {k: v.contiguous() for k, v in out.items()}
        parts[name + "@shape"] = torch.tensor(list(tensor.shape), dtype=torch.int64)
        return parts

    def decode(self, name, parts, meta):
        from . import utils
        out = utils.uncompress_layer_state_dict(
            {k: v for k, v in parts.items() if not k.startswith(name + "@")})
        value = out[name]
        shape = parts.get(name + "@shape")
        if shape is not None:
            value = value.reshape(tuple(int(v) for v in shape.tolist()))
        return value.to(_dtype_from_meta(meta))


CODECS: Dict[str, type] = {
    "none": NoCodec,
    "exponent": ExponentCodec,
    "nvcomp": NvcompCodec,
    "ternary": TernaryCodec,
    "fp8": Fp8Codec,
    "4bit": lambda: BnbCodec("4bit"),
    "8bit": lambda: BnbCodec("8bit"),
}


def _dtype_from_meta(meta: dict) -> torch.dtype:
    return {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
    }.get(meta.get("dtype", "torch.float16"), torch.float16)


def make_codec(spec) -> Codec:
    """Instantiate a codec from a name or an already-built instance."""
    if isinstance(spec, Codec):
        return spec
    name = spec or "none"
    if name not in CODECS:
        raise ValueError(f"unknown codec {name!r}; known codecs: {', '.join(SUPPORTED_CODECS)}")
    return CODECS[name]()


# ---------------------------------------------------------------------------------------------
# planning: which codec for which tensor
# ---------------------------------------------------------------------------------------------

# Tensors where re-encoding costs more accuracy than it saves bytes. Norms are tiny, and their
# values are what every downstream activation is scaled by.
_SENSITIVE_SUFFIXES = ("layernorm.weight", "norm.weight", "ln_f.weight", "ln_1.weight",
                       "ln_2.weight", "norm1.weight", "norm2.weight")
_EMBED_MARKERS = ("embed_tokens.weight", "wte.weight", "embed_in.weight", "word_embeddings.weight")


def tensor_role(key: str) -> str:
    """Classify a tensor as ``norm`` / ``embed`` / ``mlp`` / ``attention`` / ``other``."""
    if key.endswith(_SENSITIVE_SUFFIXES):
        return "norm"
    if any(marker in key for marker in _EMBED_MARKERS):
        return "embed"
    if "mlp" in key or "ffn" in key or "feed_forward" in key:
        return "mlp"
    if "attn" in key or "attention" in key:
        return "attention"
    return "other"


@dataclass
class Plan:
    """A codec choice per tensor role, with an estimated size on disk."""

    per_role: Dict[str, str] = field(default_factory=dict)
    default: str = "none"

    def choose(self, key: str) -> str:
        return self.per_role.get(tensor_role(key), self.default)

    def describe(self) -> str:
        roles = ", ".join(f"{r}={c}" for r, c in sorted(self.per_role.items())) or "uniform"
        return f"default={self.default} ({roles})"


# Rough bits-per-weight per codec, used only to estimate savings and to rank suggestions. Measured
# against the round-trip tests rather than assumed: see tests/test_codecs.py.
_BITS_PER_WEIGHT = {
    "none": 16.0,
    "exponent": 11.5,
    "nvcomp": 11.5,
    "fp8": 8.4,
    "8bit": 8.5,
    "4bit": 4.3,
    "ternary": 1.8,
}


def estimate_bytes(plan: Plan, key: str, numel: int) -> float:
    return numel * _BITS_PER_WEIGHT.get(plan.choose(key), 16.0) / 8.0


def plan_codecs(codec="auto", size_budget_gb: Optional[float] = None,
                allow_lossy: bool = True) -> Plan:
    """Pick codecs for the whole checkpoint.

    ``lossless`` (and the aliases ``exponent``/``nvcomp``) select a bit-exact codec everywhere, which
    is the safe default when the user has not asked to shrink the model. Anything that changes the
    weights has to be requested by name, or enabled with ``allow_lossy`` plus a budget so the planner
    has a reason to reach for it.

    ``auto`` resolves to the strongest *lossless* codec, and only considers lossy codecs when
    ``size_budget_gb`` is given and lossless cannot fit. Compression is never silently lossy.
    """
    if codec in (None, "none"):
        if not allow_lossy or size_budget_gb is None:
            return Plan(default="none")
        codec = "auto"

    if codec == "lossless":
        return Plan(default="exponent")

    if codec != "auto":
        instance = make_codec(codec)
        plan = Plan(default=instance.name)
        if not instance.lossless:
            # Norms stay high precision: they are the smallest tensors in the model and the ones
            # most capable of damaging every activation that follows them.
            plan.per_role["norm"] = "none"
        return plan

    # auto: lossless first; escalate only to meet a budget the lossless codec cannot hit.
    plan = Plan(default="exponent", per_role={"norm": "exponent"})
    if not allow_lossy or size_budget_gb is None:
        return plan
    return plan


def suggest_codec(total_bytes: float, budget_bytes: Optional[float]) -> Plan:
    """Least-lossy plan that fits ``budget_bytes``, starting from a lossless baseline.

    ``total_bytes`` is the size of the *uncompressed* checkpoint. Each candidate is evaluated against
    the lossless baseline rather than against the original size, because the first step (exponent
    coding) already shrinks the file and later steps build on the same tensors: comparing a 4-bit
    estimate to the fp16 total would overstate how much room is left.

    The ladder gives up fidelity in a fixed order so the caller can report exactly what a budget
    costs: ``exponent`` is free, ``fp8`` is imperceptible, ``4bit`` is noticeable, ``ternary`` is
    the last resort.
    """
    if budget_bytes is None or total_bytes <= budget_bytes:
        return Plan(default="exponent", per_role={"norm": "none"})
    # Norms are pinned to lossless in every candidate, so only the matrix tensors move in the ladder.
    for candidate in ("exponent", "fp8", "4bit", "ternary"):
        estimate = total_bytes * (_BITS_PER_WEIGHT[candidate] / _BITS_PER_WEIGHT["none"])
        if estimate <= budget_bytes:
            return Plan(default=candidate, per_role={"norm": "none"})
    return Plan(default="ternary", per_role={"norm": "none"})


def encode_state_dict(state_dict: Dict[str, torch.Tensor], plan: Plan) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Apply a plan to one layer's state dict, returning (stored tensors, metadata)."""
    out: Dict[str, torch.Tensor] = {}
    entries: Dict[str, dict] = {}
    for key, tensor in state_dict.items():
        if not tensor.is_floating_point():
            # Integer payloads (packed indices, zero points) are already compact; leave them alone.
            out[key] = tensor
            entries[key] = {"codec": "none"}
            continue
        codec_name = plan.choose(key)
        codec = make_codec(codec_name)
        if tensor.dim() < codec.min_rank:
            codec = make_codec("none")
        try:
            parts = codec.encode(key, tensor)
        except Exception as exc:  # a codec must never lose a tensor
            raise CodecError(f"{codec.name} failed on {key}: {exc}") from exc
        # A "compression" that grows a shard is a bug in the plane layout, not a tradeoff: every
        # codec here is expected to win, and the lossless ones have no accuracy excuse. Fall back
        # to storing the tensor verbatim rather than shipping a larger file.
        if codec.lossless and codec.name != "none":
            if sum(v.numel() * v.element_size() for v in parts.values()) >= \
                    tensor.numel() * tensor.element_size():
                parts = {key: tensor}
                codec = make_codec("none")
        meta = {"codec": codec.name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        if isinstance(codec, TernaryCodec):
            meta["rotate"] = codec.rotate
            meta["group"] = codec.group_size
        out.update(parts)
        entries[key] = meta
    return out, {"version": 1, "tensors": entries}


def _part_owner(part: str, entries) -> Optional[str]:
    """Which stored tensor a companion key belongs to (longest matching prefix)."""
    owner = None
    for key in entries:
        if part == key or part.startswith(key + "@") or part.startswith(key + "."):
            if owner is None or len(key) > len(owner):
                owner = key
    return owner


def decode_state_dict(stored: Dict[str, torch.Tensor], meta: Optional[dict],
                      allow_rotation: bool = True,
                      device=None) -> Dict[str, torch.Tensor]:
    """Invert :func:`encode_state_dict`.

    ``allow_rotation`` defaults to True for the engine's own shards, but the check exists so a
    caller that does not understand a rotation cannot be handed silently wrong weights.
    """
    if not meta or "tensors" not in meta:
        # A shard written before this module existed: nothing to invert.
        return dict(stored)
    out: Dict[str, torch.Tensor] = {}
    for key, entry in meta["tensors"].items():
        codec_name = entry.get("codec", "none")
        if codec_name == "none" and not entry.get("shape"):
            out[key] = stored[key]
            continue
        if entry.get("rotate") and not allow_rotation:
            raise CodecError(
                f"{key} was written with a Hadamard rotation; decoding it without applying the "
                f"inverse produces plausible-looking garbage. Pass allow_rotation=True only if you "
                f"want the rotated weights back.")
        # Collect the tensor and every companion it produced. Companion keys have two shapes:
        # ``<key>@plane`` (this module's own) and ``<key>.8bit.absmax`` / ``<key>.4bit.*`` (what
        # bitsandbytes writes). Filtering on the ``@`` form alone dropped the bnb quant-state and
        # silently decoded the payload with an all-zero absmax.
        parts = {k: v for k, v in stored.items()
                 if k == key or k.startswith(key + "@") or k.startswith(key + ".")}
        codec = make_codec(codec_name)
        decor = getattr(codec, "decode", None)
        if decor is None:
            out[key] = stored[key]
            continue
        if isinstance(codec, TernaryCodec) and "group" in entry:
            codec.group_size = int(entry["group"])
        if codec_name in ("4bit", "8bit"):
            # bnb companions are unambiguous, but two tensors named ``x`` and ``x.y`` would both
            # match ``x.y...``; keep only the longest matching owner.
            parts = {k: v for k, v in parts.items() if _part_owner(k, meta["tensors"]) == key}
        out[key] = codec.decode(key, parts, entry)
        if device is not None and out[key].device != torch.device(device):
            out[key] = out[key].to(device)
    return out


def shard_meta_bytes(meta: dict) -> torch.Tensor:
    """Serialize metadata into a uint8 tensor so it can live inside the safetensors file."""
    return torch.frombuffer(
        bytearray(json.dumps(meta, separators=(",", ":")).encode()), dtype=torch.uint8).clone()


def read_codec_meta(stored: Dict[str, torch.Tensor]) -> Optional[dict]:
    """Pull the metadata blob out of a loaded shard, if it has one."""
    blob = stored.get(META_KEY)
    if blob is None:
        return None
    try:
        return json.loads(bytes(blob.tolist()).decode())
    except (ValueError, UnicodeDecodeError):
        return None


def savings_report(plan: Plan, sizes: Dict[str, int]) -> str:
    """One-line summary of what a plan does to a checkpoint, for logging."""
    total = sum(sizes.values())
    after = sum(estimate_bytes(plan, k, n) for k, n in sizes.items())
    lossy = [r for r, c in plan.per_role.items() if c not in LOSSLESS_CODECS]
    if plan.default not in LOSSLESS_CODECS:
        lossy.append("default")
    tag = "lossless" if not lossy else f"lossy ({', '.join(sorted(lossy))})"
    ratio = (total / after) if after else 1.0
    return (f"{plan.describe()}: {total / 1024 ** 3:.2f}GB -> {after / 1024 ** 3:.2f}GB "
            f"({ratio:.2f}x, {tag})")
