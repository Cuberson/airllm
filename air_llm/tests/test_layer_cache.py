"""Unit tests for the layer cache / eviction policy.

These run on CPU and need no model download: the cache is pure accounting, and ``evict_module``
is exercised against a small module tree that mimics a decoder layer.
"""

import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / 'airllm'

if 'airllm' not in sys.modules:
    _pkg = types.ModuleType('airllm')
    _pkg.__path__ = [str(_AIRLLM_DIR)]
    sys.modules['airllm'] = _pkg

from airllm.layer_cache import StreamedLayerCache, evict_module


class TinyLayer(nn.Module):
    """A module with a nested submodule, like a real decoder layer."""

    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x):
        return self.mlp(x) + self.self_attn.q_proj(x)


def layer_nbytes(dim=8):
    return (dim * dim + dim * dim * 2) * torch.float32.itemsize


def allocated(module):
    """Bytes of *materialised* (non-meta) weight storage a module still holds.

    A meta tensor keeps its shape placeholder but owns no storage, so ``numel()`` alone would
    report bytes that are not actually resident.
    """
    return sum(p.numel() * p.element_size() for p in module.parameters()
               if p.device.type != 'meta')


class TestEviction(unittest.TestCase):
    def test_evict_module_frees_every_parameter(self):
        layer = TinyLayer()
        released = evict_module(layer)
        self.assertEqual(released, layer_nbytes())
        self.assertEqual(allocated(layer), 0)

    def test_evict_module_keeps_state_dict_keys(self):
        """A wholesale eviction must leave the module's parameter tree addressable.

        ``state_dict()`` is what checkpoints and ``tie_weights()`` walk, so an eviction that
        renames or drops parameters would break loading on the next pass.
        """
        layer = TinyLayer()
        evict_module(layer)
        self.assertEqual(sorted(layer.state_dict().keys()),
                         ['mlp.weight', 'self_attn.q_proj.weight'])

    def test_evict_subset_only_touches_named_parameters(self):
        layer = TinyLayer()
        evict_module(layer, ['mlp.weight'])
        self.assertEqual(layer.mlp.weight.numel(), 0)
        self.assertEqual(allocated(layer.self_attn), 8 * 8 * 4)

    def test_evict_unknown_name_is_ignored(self):
        layer = TinyLayer()
        self.assertEqual(evict_module(layer, ['not.a.real.param']), 0)
        self.assertEqual(allocated(layer), layer_nbytes())

    def test_evicting_twice_is_a_noop(self):
        layer = TinyLayer()
        first = evict_module(layer)
        second = evict_module(layer)
        self.assertEqual(first, layer_nbytes())
        self.assertEqual(second, 0)

    def test_evicted_module_can_be_reloaded(self):
        """An eviction must be reversible: this is what a cache miss does every token."""
        from accelerate.utils.modeling import set_module_tensor_to_device
        layer = TinyLayer()
        snapshot = {n: p.detach().clone() for n, p in layer.named_parameters()}
        evict_module(layer)
        for name, value in snapshot.items():
            set_module_tensor_to_device(layer, name, 'cpu', value=value)
        self.assertEqual(allocated(layer), layer_nbytes())
        self.assertTrue(all(p.device.type == 'cpu' for p in layer.parameters()))


class TestStreamedLayerCache(unittest.TestCase):
    def test_disabled_cache_is_inert(self):
        cache = StreamedLayerCache(gpu_budget=0, park_budget=0)
        self.assertFalse(cache.enabled)
        self.assertEqual(cache.plan(100, want_resident=True), [])
        self.assertFalse(cache.can_hold_resident(1))

    def test_resident_entries_are_bounded_by_the_gpu_budget(self):
        cache = StreamedLayerCache(gpu_budget=100)
        cache.record(0, 40, ['a'], resident=True)
        cache.record(1, 40, ['b'], resident=True)
        self.assertEqual(cache.resident_bytes, 80)
        # A third 40-byte layer needs 20 bytes freed; the LRU entry (0) is the victim.
        victims = cache.plan(40, incoming_idx=2, want_resident=True)
        self.assertEqual([v.idx for v in victims], [0])

    def test_incoming_layer_is_never_its_own_victim(self):
        cache = StreamedLayerCache(gpu_budget=10)
        cache.record(3, 10, ['x'], resident=True)
        self.assertEqual(cache.plan(10, incoming_idx=3, want_resident=True), [])

    def test_layer_larger_than_budget_is_not_cached_resident(self):
        cache = StreamedLayerCache(gpu_budget=100)
        self.assertFalse(cache.can_hold_resident(101))
        self.assertTrue(cache.can_hold_resident(100))

    def test_parked_tier_is_tracked_separately_from_vram(self):
        cache = StreamedLayerCache(gpu_budget=0, park_budget=100)
        cache.record(0, 60, ['a'], resident=False, parked={'a': torch.zeros(1)})
        self.assertEqual(cache.parked_bytes, 60)
        self.assertEqual(cache.resident_bytes, 0)
        # Evicting the parked entry frees pinned memory, not VRAM.
        self.assertEqual([v.idx for v in cache.plan(60, incoming_idx=1, want_resident=False)], [0])

    def test_mark_resident_moves_bytes_between_tiers(self):
        cache = StreamedLayerCache(gpu_budget=100, park_budget=100)
        cache.record(0, 60, ['a'], resident=False, parked={'a': torch.zeros(1)})
        cache.mark_resident(0, ['a'])
        self.assertEqual((cache.resident_bytes, cache.parked_bytes), (60, 0))

    def test_mark_evicted_moves_bytes_to_the_parked_tier(self):
        cache = StreamedLayerCache(gpu_budget=100, park_budget=100)
        cache.record(0, 60, ['a'], resident=True)
        cache.mark_evicted(0, {'a': torch.zeros(1)})
        self.assertEqual((cache.resident_bytes, cache.parked_bytes), (0, 60))
        cache.mark_evicted(0, None)  # dropping the parked copy clears its accounting
        self.assertEqual(cache.parked_bytes, 0)

    def test_lru_order_drives_victim_selection(self):
        cache = StreamedLayerCache(gpu_budget=90)
        for idx in (0, 1, 2):
            cache.record(idx, 30, [str(idx)], resident=True)
        cache.get(0)  # 0 becomes most-recently used
        victims = cache.plan(30, incoming_idx=9, want_resident=True)
        self.assertEqual([v.idx for v in victims], [1])

    def test_hits_and_misses_are_counted(self):
        cache = StreamedLayerCache(gpu_budget=100)
        cache.record(0, 10, ['a'], resident=True)
        cache.get(0)
        cache.get(7)
        stats = cache.stats()
        self.assertEqual((stats['hits'], stats['misses']), (1, 1))
        self.assertAlmostEqual(stats['hit_rate'], 0.5)

    def test_clear_returns_resident_entries_for_eviction(self):
        cache = StreamedLayerCache(gpu_budget=100)
        cache.record(0, 10, ['a'], resident=True)
        cache.record(1, 10, ['b'], resident=False, parked={'b': torch.zeros(1)})
        entries = cache.clear()
        self.assertEqual({e.idx for e in entries}, {0, 1})
        self.assertEqual((cache.resident_bytes, cache.parked_bytes), (0, 0))
        self.assertEqual(len(cache), 0)

    def test_rerecording_an_index_does_not_double_count(self):
        cache = StreamedLayerCache(gpu_budget=100)
        cache.record(0, 40, ['a'], resident=True)
        cache.record(0, 40, ['a'], resident=True)
        self.assertEqual(cache.resident_bytes, 40)
        self.assertEqual(len(cache), 1)


class TestProgress(unittest.TestCase):
    def test_throughput_reports_a_rate(self):
        import time
        from airllm.progress import Throughput
        tp = Throughput(window=10)
        self.assertEqual(tp.rate(), 0.0)
        for _ in range(5):
            tp.tick()
            time.sleep(0.005)
        self.assertGreater(tp.rate(), 0.0)
        tp.reset()
        self.assertEqual(tp.rate(), 0.0)

    def test_reporter_is_silent_when_disabled(self):
        import io
        from airllm.progress import ProgressReporter
        buf = io.StringIO()
        reporter = ProgressReporter(enabled=False, stream=buf)
        reporter.begin('load')
        reporter.beat('x', force=True)
        reporter.end()
        self.assertEqual(buf.getvalue(), '')

    def test_reporter_emits_stage_lines_when_enabled(self):
        import io
        from airllm.progress import ProgressReporter
        buf = io.StringIO()
        reporter = ProgressReporter(enabled=True, stream=buf)
        reporter.begin('split', 'model')
        reporter.end('split')
        out = buf.getvalue()
        self.assertIn('split: model', out)
        self.assertIn('split finished', out)

    def test_stall_guard_stays_quiet_for_fast_work(self):
        from airllm.progress import stall_guard
        with stall_guard(5, 'quick'):
            pass

    def test_stall_guard_raises_when_the_window_passes(self):
        from airllm.progress import StallError, stall_guard
        with self.assertRaises(StallError):
            with stall_guard(0.05, 'slow thing'):
                import time
                time.sleep(0.12)


if __name__ == '__main__':
    unittest.main()
