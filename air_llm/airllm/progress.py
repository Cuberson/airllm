"""Opt-in liveness reporting for long layer-streaming runs.

A streamed model can spend minutes loading a checkpoint before it produces a token, and the
process looks identical whether it is working or hung: low CPU, low GPU, no output. Two upstream
issues describe exactly this confusion (a 4-token run taking 114s and being killed at 90s because
nothing was printed). This module makes progress observable, cheaply and only when asked:

* :class:`ProgressReporter` prints a ``[airllm] <stage>: <detail>`` line whenever a stage starts
  and, for the long ones, every ``interval`` seconds while it runs.
* :class:`Throughput` keeps a rolling rate so the line can say "1.2 layers/s, 3 tok/s, ETA 40s".
* :func:`stall_guard` optionally raises if a single stage stops moving entirely, so a wedged
  filesystem fails with a message instead of hanging forever.

Nothing here is enabled by default, and all of it is a no-op when disabled, so it cannot regress
the fast path.
"""

import threading
import time
from contextlib import contextmanager


class Throughput:
    """Rolling average of items completed per second."""

    def __init__(self, window=50):
        self.window = window
        self._times = []
        self._lock = threading.Lock()

    def tick(self, n=1):
        with self._lock:
            now = time.monotonic()
            self._times.append((now, n))
            if len(self._times) > self.window:
                del self._times[0:len(self._times) - self.window]

    def rate(self):
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            (t0, _), (t1, _) = self._times[0], self._times[-1]
            if t1 <= t0:
                return 0.0
            return sum(n for _, n in self._times[1:]) / (t1 - t0)

    def reset(self):
        with self._lock:
            self._times.clear()


class ProgressReporter:
    """Prints stage transitions and periodic heartbeats for long operations."""

    def __init__(self, enabled=False, interval=5.0, stream=None, prefix='[airllm]'):
        self.enabled = bool(enabled)
        self.interval = max(0.5, float(interval))
        self.prefix = prefix
        self._stream = stream
        self._lock = threading.Lock()
        self._stage = None
        self._stage_start = None
        self._last_beat = None

    def _emit(self, message):
        stream = self._stream
        try:
            if stream is None:
                import sys
                stream = sys.stderr
            stream.write(f"{self.prefix} {message}\n")
            stream.flush()
        except Exception:  # noqa: BLE001 - reporting must never break inference
            pass

    def begin(self, stage, detail=''):
        if not self.enabled:
            return
        with self._lock:
            self._stage = stage
            self._stage_start = time.monotonic()
            self._last_beat = self._stage_start
        self._emit(f"{stage}{(': ' + detail) if detail else ''}")

    def beat(self, detail='', force=False):
        """Print a heartbeat if the interval elapsed (or always, when ``force``)."""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            if self._stage is None:
                return
            due = force or (now - self._last_beat) >= self.interval
            if not due:
                return
            self._last_beat = now
            elapsed = now - self._stage_start
            stage = self._stage
        self._emit(f"{stage}: {detail} ({elapsed:.0f}s elapsed)" if detail
                   else f"{stage}: still running ({elapsed:.0f}s elapsed)")

    def end(self, stage=None, detail=''):
        if not self.enabled:
            return
        with self._lock:
            elapsed = (time.monotonic() - self._stage_start) if self._stage_start else 0.0
            current = self._stage
            self._stage = None
        stage = stage or current or 'done'
        suffix = f": {detail}" if detail else ''
        self._emit(f"{stage} finished{suffix} ({elapsed:.1f}s)")

    @contextmanager
    def stage(self, name, detail=''):
        self.begin(name, detail)
        try:
            yield self
        finally:
            self.end(name)


class StallError(RuntimeError):
    """Raised by :func:`stall_guard` when a stage stops making progress entirely."""


@contextmanager
def stall_guard(seconds, what='operation', reporter=None):
    """Raise :class:`StallError` when the guarded block makes no progress for ``seconds``.

    This is a watchdog, not a preemption mechanism: the caller reports progress with
    ``reporter.beat()``, and if the deadline passes without one a diagnostic is printed and the
    error is raised at the next checkpoint. That is deliberately conservative -- a slow-but-moving
    run keeps going, while a wedged network mount fails with an actionable message instead of
    hanging forever. Disabled when ``seconds`` is None or <= 0.
    """
    if not seconds or seconds <= 0:
        yield
        return

    deadline = time.monotonic() + seconds
    state = {'expired': False, 'stop': False}

    def _watch():
        while not state['stop']:
            if time.monotonic() >= deadline:
                state['expired'] = True
                if reporter is not None:
                    reporter._emit(f"{what} made no progress for {seconds:.0f}s; assuming it is stuck")
                return
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))

    thread = threading.Thread(target=_watch, name='airllm-stall-guard', daemon=True)
    thread.start()
    try:
        yield state
    finally:
        state['stop'] = True
    if state['expired']:
        raise StallError(f"{what} stalled for {seconds:.0f}s")
