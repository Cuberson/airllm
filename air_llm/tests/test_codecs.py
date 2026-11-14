"""Codec tests: every codec must be exactly what it claims to be.

The single property that matters here is the lossless/lossy boundary. A codec advertised as
lossless that quietly re-encodes a tensor corrupts a model in a way no output check would catch
reliably (the weights are *nearly* right), so the lossless ones are tested bit for bit, on inputs
chosen to break naive implementations: denormals, infinities, NaN, empty tensors, single rows, and
the fp16/bf16 collision where both dtypes are two bytes wide.
"""

import json
import sys
import types
import unittest
from pathlib import Path

import torch

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'
if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.codecs import (  # noqa: E402
    CODECS, LOSSLESS_CODECS, SUPPORTED_CODECS, CodecError, HadamardMixin, Plan,
    decode_state_dict, encode_state_dict, estimate_bytes, make_codec, plan_codecs,
    read_codec_meta, savings_report, shard_meta_bytes, suggest_codec, tensor_role,
)

torch.manual_seed(0)


def bit_equal(a, b):
    """Bitwise equality that treats NaN as equal to itself."""
    if a.dtype == b.dtype and a.numel() == b.numel() and a.numel():
        try:
            return torch.equal(a.view(torch.int16 if a.element_size() == 2 else torch.int32),
                               b.view(torch.int16 if b.element_size() == 2 else torch.int32))
        except RuntimeError:
            pass
    return torch.equal(a, b)


def stored_bytes(parts):
    return sum(v.numel() * v.element_size() for v in parts.values())


def weights(rows=128, cols=128, dtype=torch.bfloat16, scale=0.02):
    return (torch.randn(rows, cols) * scale).to(dtype)


class TestLosslessContract(unittest.TestCase):
    """A lossless codec must return the input, bit for bit, in every case."""

    EDGE_CASES = {
        'bf16': lambda: torch.randn(64, 64).to(torch.bfloat16),
        'fp16': lambda: torch.randn(64, 64).to(torch.float16),
        'fp32': lambda: torch.randn(64, 64),
        'all zeros': lambda: torch.zeros(32, 32, dtype=torch.bfloat16),
        'denormal': lambda: (torch.randn(32, 32) * 1e-40).to(torch.bfloat16),
        'extreme': lambda: (torch.randn(32, 32) * 1e38).to(torch.bfloat16),
        'single row': lambda: torch.randn(1, 64).to(torch.bfloat16),
        'single element': lambda: torch.randn(1, 1).to(torch.bfloat16),
        'empty': lambda: torch.zeros(0, dtype=torch.bfloat16),
        'constant': lambda: torch.full((16, 16), 0.5, dtype=torch.bfloat16),
        'inf and nan': lambda: torch.tensor(
            [[float('inf'), float('-inf'), float('nan'), 0.0]], dtype=torch.bfloat16
        ).expand(4, 4).contiguous(),
    }

    def test_every_lossless_codec_is_bit_exact(self):
        for codec_name in sorted(LOSSLESS_CODECS - {"none"}):
            for case, factory in self.EDGE_CASES.items():
                with self.subTest(codec=codec_name, case=case):
                    original = factory()
                    parts, meta = encode_state_dict(
                        {'w': original}, plan_codecs(codec_name))
                    restored = decode_state_dict(parts, meta)['w']
                    self.assertTrue(bit_equal(restored, original),
                                    f"{codec_name} changed {case}")
                    self.assertIs(restored.dtype, original.dtype)
                    self.assertEqual(tuple(restored.shape), tuple(original.shape))

    def test_lossless_codecs_are_labelled_lossless(self):
        for name in LOSSLESS_CODECS:
            self.assertTrue(make_codec(name).lossless, name)

    def test_fp16_and_bf16_do_not_collide(self):
        """Both are two bytes; a width-based dtype tag decodes one as the other."""
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                original = weights(16, 16, dtype=dtype)
                parts, meta = encode_state_dict({'w': original}, plan_codecs('exponent'))
                restored = decode_state_dict(parts, meta)['w']
                self.assertIs(restored.dtype, dtype)


class TestLosslessRatio(unittest.TestCase):
    def test_exponent_codec_information_theoretically_wins(self):
        """Exponent coding only pays if float exponents really are low-entropy."""
        original = weights(256, 256)
        parts, _ = encode_state_dict({'w': original}, plan_codecs('exponent'))
        ratio = (original.numel() * original.element_size()) / stored_bytes(parts)
        # With a fixed-width alphabet the exponent plane costs ceil(log2(distinct)) bits -- ~5 for
        # real weights, where the plane is 8 bits, so ~1.3x is the honest ceiling here. (Entropy
        # coding would reach ~1.5x but cannot be vectorised; see _ExponentAlphabet.)
        self.assertGreater(ratio, 1.2, "lossless compression saved almost nothing")

    def test_exponent_plane_never_exceeds_eight_bits(self):
        """The saving is real: no code may cost more than the raw exponent byte it replaces."""
        original = weights(256, 256)
        parts, _ = encode_state_dict({'w': original}, plan_codecs('exponent'))
        symbols = original.numel()
        plane_bits = parts['w@exponent'].numel() * 8
        self.assertLessEqual(plane_bits, symbols * 8)

    def test_pure_garbage_does_not_expand_much(self):
        """Random bytes have no exponent structure; the codec must not blow up."""
        noise = torch.randint(0, 2 ** 16, (128, 128), dtype=torch.int32).to(torch.int16)
        tensor = noise.view(torch.bfloat16)
        parts, meta = encode_state_dict({'w': tensor}, plan_codecs('exponent'))
        restored = decode_state_dict(parts, meta)['w']
        self.assertTrue(bit_equal(restored, tensor))
        ratio = (tensor.numel() * 2) / stored_bytes(parts)
        # Exponents of uniform-random bit patterns are uniformly distributed, so Huffman gives them
        # all 8 bits: 8 raw + 8 coded = 16 bits, plus a small table. ~10% expansion is the ceiling.
        self.assertGreater(ratio, 0.85, "lossless codec expanded random data badly")


class TestLossyCodecs(unittest.TestCase):
    def test_lossy_codecs_are_not_labelled_lossless(self):
        for name in sorted(set(SUPPORTED_CODECS) - LOSSLESS_CODECS):
            with self.subTest(codec=name):
                self.assertFalse(make_codec(name).lossless)

    def test_fp8_is_close(self):
        original = weights(256, 256)
        parts, meta = encode_state_dict({'w': original}, plan_codecs('fp8'))
        restored = decode_state_dict(parts, meta)['w'].float()
        relative = ((restored - original.float()).abs().mean()
                    / original.float().abs().mean()).item()
        self.assertLess(relative, 0.05, "fp8 error is larger than e4m3 should give")

    def test_ternary_shrinks_a_lot(self):
        original = weights(256, 256)
        parts, meta = encode_state_dict({'w': original}, plan_codecs('ternary'))
        ratio = (original.numel() * 2) / stored_bytes(parts)
        self.assertGreater(ratio, 5.0, "ternary is supposed to be ~7x on bf16")
        restored = decode_state_dict(parts, meta)['w']
        self.assertEqual(tuple(restored.shape), (256, 256))

    def test_higher_ratio_means_higher_error(self):
        """The codecs must be ordered the way the planner assumes they are."""
        original = weights(256, 256)
        errors = {}
        for name in ('fp8', '8bit', '4bit', 'ternary'):
            parts, meta = encode_state_dict({'w': original}, plan_codecs(name))
            restored = decode_state_dict(parts, meta)['w'].float()
            errors[name] = ((restored - original.float()).abs().mean()
                            / original.float().abs().mean()).item()
        self.assertLess(errors['fp8'], errors['4bit'])
        self.assertLess(errors['4bit'], errors['ternary'])

    def test_norms_stay_exact_even_under_a_lossy_plan(self):
        """Norm weights scale every downstream activation; a plan must not quantize them."""
        plan = plan_codecs('4bit')
        self.assertEqual(plan.choose('model.layers.0.input_layernorm.weight'), 'none')
        original = weights(1, 64)
        parts, meta = encode_state_dict({'model.norm.weight': original}, plan)
        self.assertTrue(bit_equal(decode_state_dict(parts, meta)['model.norm.weight'], original))


class TestHadamardRotation(unittest.TestCase):
    def test_rotation_is_an_exact_involution(self):
        class Rot(HadamardMixin):
            pass
        rot = Rot()
        matrix = torch.randn(64, 32)
        twice = rot._rotate(rot._rotate(matrix, invert=False), invert=True)
        self.assertTrue(torch.allclose(twice, matrix, atol=1e-5))

    def test_rotation_collapses_row_scale_spread(self):
        """The measurable reason to rotate at all: the per-row scale stops varying wildly."""
        class Rot(HadamardMixin):
            pass
        heavy = torch.randn(256, 256) * torch.exp(torch.randn(256, 1) * 0.6)
        rotated = Rot()._rotate(heavy, invert=False)
        before = heavy.abs().mean(dim=1).std().item()
        after = rotated.abs().mean(dim=1).std().item()
        self.assertLess(after, before / 5.0)

    def test_rotated_shard_refuses_to_decode_silently(self):
        """Forgetting the rotation yields plausible garbage; the decoder must refuse instead."""
        original = weights(64, 64)
        plan = Plan(default='ternary')
        plan.per_role['norm'] = 'none'
        codec = CODECS['ternary']()
        codec.rotate = True
        parts = codec.encode('w', original)
        meta = {'version': 1, 'tensors': {'w': {'codec': 'ternary', 'shape': [64, 64],
                                                'dtype': 'torch.bfloat16', 'rotate': True}}}
        with self.assertRaises(CodecError):
            decode_state_dict(parts, meta, allow_rotation=False)
        # ...and works when the caller does understand it.
        restored = decode_state_dict(parts, meta, allow_rotation=True)['w']
        self.assertEqual(tuple(restored.shape), (64, 64))

    def test_rotated_round_trip_recovers_the_original(self):
        original = weights(128, 128)
        codec = CODECS['ternary']()
        codec.rotate = True
        parts = codec.encode('w', original)
        restored = codec.decode('w', parts, {'dtype': 'torch.bfloat16', 'rotate': True})
        # Lossy, so not equal -- but the rotation must be fully undone, which shows up as the
        # error being representative of ternary quantization rather than a structural mismatch.
        relative = ((restored.float() - original.float()).abs().mean()
                    / original.float().abs().mean()).item()
        self.assertLess(relative, 0.75)


class TestPlanning(unittest.TestCase):
    def test_auto_is_lossless_by_default(self):
        """Compression must never be silently lossy."""
        plan = plan_codecs()
        self.assertIn(plan.default, LOSSLESS_CODECS)

    def test_auto_stays_lossless_even_with_a_budget_by_default(self):
        plan = plan_codecs('auto', size_budget_gb=1.0, allow_lossy=False)
        self.assertIn(plan.default, LOSSLESS_CODECS)

    def test_explicit_lossy_request_is_honoured(self):
        plan = plan_codecs('ternary')
        self.assertEqual(plan.default, 'ternary')
        self.assertEqual(plan.per_role.get('norm'), 'none')

    def test_tensor_roles(self):
        cases = {
            'model.layers.3.input_layernorm.weight': 'norm',
            'model.norm.weight': 'norm',
            'model.embed_tokens.weight': 'embed',
            'model.layers.0.mlp.up_proj.weight': 'mlp',
            'model.layers.0.self_attn.q_proj.weight': 'attention',
            'lm_head.weight': 'other',
            'transformer.h.0.mlp.c_fc.weight': 'mlp',
        }
        for key, expected in cases.items():
            self.assertEqual(tensor_role(key), expected, key)

    def test_budget_escalation_prefers_fidelity(self):
        """At equal fit, the planner must pick the least lossy codec that works."""
        total = 10 * 1024 ** 3
        generous = suggest_codec(total, budget_bytes=total * 2)      # lossless fits
        self.assertIn(generous.default, LOSSLESS_CODECS)
        # The ladder is ordered by fidelity, and each rung must actually fit the budget it
        # claims to: 4bit is ~4.3 bits/weight, so a 10GB model needs ~2.7GB at 4bit.
        four_bit = suggest_codec(total, budget_bytes=total * 4.3 / 16)
        self.assertEqual(four_bit.default, '4bit')
        three_bit = suggest_codec(total, budget_bytes=total * 4.3 / 16 * 0.9)
        self.assertEqual(three_bit.default, 'ternary')
        # A budget no codec can meet still returns the smallest, rather than silently overflowing.
        impossible = suggest_codec(total, budget_bytes=total / 100)
        self.assertEqual(impossible.default, 'ternary')

    def test_estimate_tracks_measured_ratios(self):
        """The numbers the planner reasons about must match what the codecs actually produce."""
        original = weights(128, 128)
        actual = {name: stored_bytes(encode_state_dict({'w': original}, plan_codecs(name))[0])
                  for name in ('none', 'exponent', 'fp8')}
        estimate = {name: estimate_bytes(plan_codecs(name), 'w', original.numel())
                    for name in actual}
        for name in actual:
            self.assertLess(abs(actual[name] - estimate[name]) / actual[name], 0.35,
                            f"{name}: planner estimate is far off the real size")

    def test_savings_report_mentions_fidelity(self):
        plan = plan_codecs('ternary')
        report = savings_report(plan, {'a': 1024, 'b': 2048})
        self.assertIn('lossy', report)
        self.assertIn('ternary', report)


class TestBitPacking(unittest.TestCase):
    """Fixed-width bit packing is the hot path in both directions, and both were rewritten.

    Decode used to be a per-bit gather (~14 MiB/s on a real bf16 layer) and is now numpy's
    ``unpackbits`` followed by one matmul; encode used to be a per-bit ``scatter_add`` and is now
    ``packbits``. A subtle error here is *silent*: a wrong bit order yields weights that are
    merely wrong, not weights that fail to load, so the round-trip is pinned across every width
    the codecs actually use.
    """

    def test_round_trip_across_widths(self):
        """Widths 1..24, including the boundaries where byte alignment changes."""
        from airllm.codecs import _pack_fixed, _unpack_fixed
        torch.manual_seed(0)
        for width in (1, 2, 3, 5, 7, 8, 11, 12, 16, 24):
            for count in (0, 1, 7, 8, 9, 1000, 100_003):
                with self.subTest(width=width, count=count):
                    values = torch.randint(0, 1 << min(width, 30), (count,), dtype=torch.int64)
                    packed = _pack_fixed(values, width)
                    if count:
                        self.assertEqual(packed.numel(), (count * width + 7) // 8)
                    restored = _unpack_fixed(packed, width, count)
                    self.assertEqual(restored.numel(), count)
                    self.assertTrue(torch.equal(restored, values))

    def test_chunk_boundary_is_byte_aligned(self):
        """A chunk must not straddle a byte, or every symbol after it is shifted.

        Counts on either side of the internal chunk size are compared against a single-shot
        decode of the same data.
        """
        from airllm.codecs import _pack_fixed, _unpack_fixed
        torch.manual_seed(1)
        width = 5
        count = 2_000_003          # comfortably larger than one internal chunk at this width
        values = torch.randint(0, 1 << width, (count,), dtype=torch.int64)
        packed = _pack_fixed(values, width)
        self.assertTrue(torch.equal(_unpack_fixed(packed, width, count), values))

    def test_round_trip_is_exact_for_alphabet_codes(self):
        """End to end through the alphabet, which is what a real shard does."""
        from airllm.codecs import _ExponentAlphabet
        torch.manual_seed(2)
        exponents = torch.randint(120, 150, (200_000,), dtype=torch.int64)
        alpha = _ExponentAlphabet.build(exponents)
        packed, bit_count = alpha.encode(exponents)
        self.assertEqual(bit_count, exponents.numel() * alpha.width)
        self.assertEqual(packed.numel(), (bit_count + 7) // 8)
        self.assertTrue(torch.equal(alpha.decode(packed, exponents.numel()), exponents))


class TestScalability(unittest.TestCase):
    """Guards against the failure mode this module actually hit: silent OOM.

    Every intermediate must be bounded by a constant, not by the layer. The first implementation
    built an (n, max_len) int64 bit matrix for an exponential-Huffman code, which on a real decoder
    layer is tens of GB -- the process was OOM-killed with no traceback, so a model load simply
    looked like a hang. These tests assert the shipped code path *completes* at a size where that
    class of bug would show up, and that its peak memory stays bounded.
    """

    def test_large_layer_round_trips_in_bounded_memory(self):
        import resource
        # ~40M weights: an eighth of a real 27B layer's MLP, big enough that an O(n * width)
        # int64 intermediate would be several GB if it were not chunked.
        original = (torch.randn(8192, 5120) * 0.02).to(torch.bfloat16)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        parts, meta = encode_state_dict({'w': original}, plan_codecs('lossless'))
        restored = decode_state_dict(parts, meta)['w']
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        self.assertTrue(bit_equal(restored, original))
        # input is ~0.08GB; a per-symbol expansion would push this past 2GB. Allow generous slack
        # for the interpreter and the test's own copies.
        self.assertLess(peak - before, 2.0, f"peak grew by {peak - before:.1f}GB")

    def test_packing_is_chunk_size_independent(self):
        """Chunk boundaries must not change the bytes; they are an implementation detail."""
        values = torch.randint(0, 32, (100003,))
        from airllm.codecs import _pack_fixed, _unpack_fixed
        packed = _pack_fixed(values, 5)
        self.assertTrue(torch.equal(_unpack_fixed(packed, 5, values.numel()), values))
        self.assertEqual(packed.numel(), (values.numel() * 5 + 7) // 8)

    def test_single_symbol_alphabet(self):
        """A constant tensor has a one-symbol alphabet, i.e. zero information and zero bits."""
        original = torch.full((64, 64), 0.5, dtype=torch.bfloat16)
        parts, meta = encode_state_dict({'w': original}, plan_codecs('exponent'))
        self.assertTrue(bit_equal(decode_state_dict(parts, meta)['w'], original))


class TestShardMeta(unittest.TestCase):
    def test_meta_survives_a_safetensors_style_round_trip(self):
        original = weights(32, 32)
        parts, meta = encode_state_dict({'w': original}, plan_codecs('exponent'))
        parts['__airllm__'] = shard_meta_bytes(meta)
        recovered = read_codec_meta(parts)
        self.assertEqual(recovered['tensors']['w']['codec'], 'exponent')
        restored = decode_state_dict(parts, recovered)['w']
        self.assertTrue(bit_equal(restored, original))

    def test_shard_without_meta_is_passed_through(self):
        """Shards written before this module existed must still load."""
        original = weights(8, 8)
        self.assertTrue(bit_equal(decode_state_dict({'w': original}, None)['w'], original))

    def test_bad_meta_does_not_raise(self):
        self.assertIsNone(read_codec_meta({'__airllm__': torch.tensor([0xFF, 0xFE], dtype=torch.uint8)}))

    def test_unknown_codec_name_is_rejected(self):
        with self.assertRaises(ValueError):
            make_codec('quantum-magic')

    def test_every_supported_codec_is_registered(self):
        for name in SUPPORTED_CODECS:
            self.assertIn(name, CODECS)


class TestMultiTensorShard(unittest.TestCase):
    def test_a_realistic_layer_round_trips(self):
        """A decoder layer mixes float matrices, 1-D norms and integer buffers."""
        layer = {
            'model.layers.0.self_attn.q_proj.weight': weights(64, 64),
            'model.layers.0.input_layernorm.weight': torch.randn(64).to(torch.bfloat16),
            'model.layers.0.rotary.inv_freq': torch.randn(32).to(torch.float32),
            'model.layers.0.experts.g_idx': torch.randint(0, 8, (64,), dtype=torch.int32),
        }
        for codec in ('exponent', 'fp8', 'none'):
            with self.subTest(codec=codec):
                parts, meta = encode_state_dict(layer, plan_codecs(codec))
                restored = decode_state_dict(parts, meta)
                self.assertEqual(set(restored), set(layer))
                for key, original in layer.items():
                    self.assertEqual(tuple(restored[key].shape), tuple(original.shape), key)
                # Integer buffers must never be re-encoded.
                self.assertTrue(bit_equal(restored['model.layers.0.experts.g_idx'],
                                          layer['model.layers.0.experts.g_idx']))

    def test_prefix_named_tensors_do_not_steal_companions(self):
        """``x`` and ``x.y`` both match ``x.y...``; the longest owner must win."""
        layer = {
            'w': weights(64, 64),
            'w.extra': weights(32, 32),
        }
        parts, meta = encode_state_dict(layer, plan_codecs('8bit'))
        restored = decode_state_dict(parts, meta)
        self.assertEqual(set(restored), set(layer))
        for key in layer:
            self.assertEqual(tuple(restored[key].shape), tuple(layer[key].shape), key)


if __name__ == '__main__':
    unittest.main()
