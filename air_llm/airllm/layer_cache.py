"""Layer caching and eviction for AirLLM's weight-streaming engine.

AirLLM runs a model that is far larger than VRAM by materialising exactly one module at a time:
a forward pre-hook copies the module's shard disk -> (host) -> device, and a forward post-hook
sends it back to the ``meta`` device so the next module can reuse the memory.

Two things dominate the cost of that loop, and both are avoidable:

1. **Re-reading the same bytes for every generated token.** A 70B model touches ~140GB of shards
   per token. Any shard that is still in VRAM, or still in pinned host memory, does not have to be
   read again. :class:`StreamedLayerCache` is an LRU over streamed modules that keeps the hottest
   ones in one of two places:

   * *resident*: the weights stay loaded in the module (VRAM). A hit is free, but it costs VRAM.
   * *parked*: the weights go back to ``meta`` but the pinned host copy is kept. A hit costs one
     host -> device copy, i.e. the part of the load that was already fast, and skips the disk read.

   Both tiers are byte-budgeted, so a machine never caches its way into an OOM.

2. **Paying a full-process garbage collection per module.** ``gc.collect()`` plus
   ``cuda.empty_cache()`` costs tens of milliseconds on a large Python heap, which is often more
   than loading the layer. :func:`evict_parameters` releases weights without those calls, and the
   engine only flushes the allocator when it actually needs the memory back.

Everything here is deliberately dependency-free (no GPU required to import) so the policy can be
unit-tested on CPU.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from accelerate.utils.modeling import set_module_tensor_to_device


# ---------------------------------------------------------------------------------------------
# eviction
# ---------------------------------------------------------------------------------------------

def evict_module(module, param_names=None) -> int:
    """Release a module's weights (or just ``param_names`` of them) to ``meta``.

    When every parameter is released this is a single ``module.to('meta')``, which is both the
    cheapest option (~6us for a decoder layer) and the most robust: it keeps the Parameter objects
    alive, so ``state_dict`` keys, tie_weights, and hook references all stay valid.

    ``param_names`` narrows the release to a subset (a MoE expert, or the tensors one hook placed).
    Those are matched on module-local suffixes, since the caller owns only the submodule.
    """
    params = list(module.parameters(recurse=True))
    if not params:
        return 0

    if param_names is None or set(param_names) >= {n for n, _ in module.named_parameters(recurse=True)}:
        released = sum(p.numel() * p.element_size() for p in params if p.device.type != 'meta')
        module.to('meta')
        return released

    release = set(param_names)
    released = 0
    for name, param in list(module.named_parameters(recurse=True)):
        if name not in release or param.device.type == 'meta':
            continue
        released += param.numel() * param.element_size()
        # Shrink the storage in place: keeps the Parameter object (so state_dict keys and
        # tie_weights stay valid) and frees the device block without a solver round-trip.
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    return released


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """One recently used streamed module.

    ``moved`` is the list of parameter names the load hook materialised, which is exactly what an
    eviction has to undo. ``parked`` holds the (pinned) host copy when the module has been sent
    back to ``meta``; ``resident`` means the weights are still in the module itself.
    """

    idx: int
    nbytes: int
    moved: List[str] = field(default_factory=list)
    parked: Optional[Dict[str, torch.Tensor]] = None
    resident: bool = False
    # Modules that are cheap to reload (small norms, the final projection) should not evict a
    # multi-GB decoder layer: the layer's miss costs a disk read, the small module's costs almost
    # nothing. ``priority`` is a per-module load-cost estimate the engine fills in.
    priority: float = 1.0

    @property
    def pinned(self) -> bool:
        """True when the parked copy is page-locked (so the restore can be a direct DMA)."""
        if not self.parked:
            return False
        try:
            return all(t.is_pinned() for t in self.parked.values() if isinstance(t, torch.Tensor))
        except RuntimeError:
            return False


class StreamedLayerCache:
    """Byte-budgeted LRU over streamed modules, with a VRAM tier and a pinned-host tier.

    Two independent budgets are tracked:

    ``gpu_budget``
        How many bytes may stay *materialised on the device*. The engine sizes this so that the
        peak (the layer being loaded plus everything resident) still fits in VRAM, and it never
        exceeds it: a single layer larger than the budget simply is not cached.

    ``park_budget``
        How many bytes of pinned host copies may be kept after eviction. This tier costs no VRAM,
        so it stays useful on a 4GB card where the GPU tier is disabled. Pinned memory cannot be
        swapped out, so the default is deliberately small and configurable.

    Entries are ordered least- to most-recently used. ``plan`` returns the entries a caller must
    release before it can hold ``incoming_bytes`` more, which is the only place policy lives.
    """

    def __init__(self, gpu_budget: int = 0, park_budget: int = 0):
        self.gpu_budget = max(0, int(gpu_budget))
        self.park_budget = max(0, int(park_budget))
        self._entries: 'OrderedDict[int, CacheEntry]' = OrderedDict()
        # Running totals, per tier. ``resident_bytes`` counts entries whose weights are in VRAM.
        self.resident_bytes = 0
        self.parked_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -- queries ------------------------------------------------------------------------------

    def __contains__(self, idx: int) -> bool:
        return idx in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return self.gpu_budget > 0 or self.park_budget > 0

    def get(self, idx: int) -> Optional[CacheEntry]:
        """Return the entry for ``idx`` (marking it most-recently used) or None."""
        entry = self._entries.get(idx)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(idx)
        self.hits += 1
        return entry

    def peek(self, idx: int) -> Optional[CacheEntry]:
        """Like :meth:`get` but does not count as a hit (used for prefetch planning)."""
        return self._entries.get(idx)

    def can_hold_resident(self, nbytes: int) -> bool:
        """Whether a module of ``nbytes`` fits in the VRAM tier at all."""
        return self.gpu_budget > 0 and nbytes <= self.gpu_budget

    def can_park(self, nbytes: int) -> bool:
        return self.park_budget > 0 and nbytes <= self.park_budget

    # -- mutation -----------------------------------------------------------------------------

    def record(self, idx: int, nbytes: int, moved: List[str], *, resident: bool,
               parked: Optional[Dict[str, torch.Tensor]] = None,
               priority: float = 1.0) -> None:
        """Register (or refresh) an entry. Call :meth:`plan` first to make room."""
        self.discard(idx, release=False, count_eviction=False)
        entry = CacheEntry(idx=idx, nbytes=nbytes, moved=list(moved),
                           parked=parked, resident=resident, priority=priority)
        self._entries[idx] = entry
        self._entries.move_to_end(idx)
        if resident:
            self.resident_bytes += nbytes
        elif parked:
            self.parked_bytes += nbytes

    def plan(self, incoming_bytes: int, incoming_idx: Optional[int] = None,
             *, want_resident: bool) -> List[CacheEntry]:
        """Entries that must be released before ``incoming_bytes`` more can be held.

        ``want_resident`` says which tier the incoming module is headed for, so only that tier's
        budget needs to be respected. The incoming module itself is never a victim.
        """
        if not self.enabled:
            return []
        budget = self.gpu_budget if want_resident else self.park_budget
        used = self.resident_bytes if want_resident else self.parked_bytes
        if budget <= 0 or used + incoming_bytes <= budget:
            return []

        # Cheapest-first: among the entries that can be dropped, prefer the ones whose reload is
        # least painful (lowest priority, then smallest), breaking ties by age. A single large
        # decoder layer is then only evicted when nothing cheaper is left, instead of being the
        # automatic LRU victim of a small norm or the output head.
        candidates = [(entry.priority, entry.nbytes, pos, entry)
                      for pos, (idx, entry) in enumerate(self._entries.items())
                      if idx != incoming_idx]
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        victims = []
        for _priority, nbytes, _pos, entry in candidates:
            if used + incoming_bytes <= budget:
                break
            victims.append(entry)
            used -= nbytes
        # Keep them in LRU order so the engine evicts oldest-first when it walks the list.
        order = {e.idx: i for i, e in enumerate(self._entries.values())}
        victims.sort(key=lambda e: order[e.idx])
        return victims

    def discard(self, idx: int, *, release: bool = True, count_eviction: bool = True) -> None:
        """Forget ``idx``. ``release`` says whether an still-resident entry must be evicted first."""
        entry = self._entries.pop(idx, None)
        if entry is None:
            return
        if entry.resident:
            self.resident_bytes -= entry.nbytes
        elif entry.parked:
            self.parked_bytes -= entry.nbytes
        if count_eviction:
            self.evictions += 1
        if release and entry.resident:
            entry.resident = False

    def mark_evicted(self, idx: int, parked: Optional[Dict[str, torch.Tensor]]) -> None:
        """Update an entry after its weights left the module (parked copy or dropped)."""
        entry = self._entries.get(idx)
        if entry is None:
            return
        if entry.resident:
            self.resident_bytes -= entry.nbytes
            entry.resident = False
        if entry.parked is not None:
            self.parked_bytes -= entry.nbytes
            entry.parked = None
        if parked:
            entry.parked = parked
            self.parked_bytes += entry.nbytes
        entry.moved = []

    def mark_resident(self, idx: int, moved: List[str]) -> None:
        """Update an entry after its weights were re-materialised from the parked copy."""
        entry = self._entries.get(idx)
        if entry is None:
            return
        if entry.parked:
            self.parked_bytes -= entry.nbytes
            entry.parked = None
        if not entry.resident:
            self.resident_bytes += entry.nbytes
            entry.resident = True
        entry.moved = list(moved)
        self._entries.move_to_end(idx)

    def clear(self) -> List[CacheEntry]:
        """Drop every entry, returning the ones the caller still has to evict."""
        entries = list(self._entries.values())
        self._entries.clear()
        self.resident_bytes = 0
        self.parked_bytes = 0
        return entries

    # -- reporting ----------------------------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            'entries': len(self._entries),
            'resident_mb': self.resident_bytes / 1024 ** 2,
            'parked_mb': self.parked_bytes / 1024 ** 2,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (self.hits / total) if total else 0.0,
        }

    def describe(self) -> str:
        s = self.stats()
        return (f"layer cache: {s['entries']} entries, {s['resident_mb']:.0f}MB resident, "
                f"{s['parked_mb']:.0f}MB parked, hit rate {s['hit_rate'] * 100:.0f}% "
                f"({s['hits']}/{s['hits'] + s['misses']})")
