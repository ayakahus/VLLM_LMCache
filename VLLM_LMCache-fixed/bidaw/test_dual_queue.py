#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline sanity checks for bidaw.dual_queue_scheduler (no server needed)."""
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bidaw.dual_queue_scheduler import STATS, apply_dual_queue_patch  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))


# ---------------------------------------------------------------------
# 1. HRRN dispatch order with a fake controller
# ---------------------------------------------------------------------
class FakeGroup:
    def __init__(self, n):
        self.keys = list(range(n))


class FakeCtrl:
    """Mimics the PrefetchController surface the patch touches."""

    def __init__(self, items):
        self._submission_lock = threading.Lock()
        self._pending_queue = deque(items)
        self._bidaw_enqueue_monotonic = {}
        self._in_flight_requests = {}
        self.started = []

    def _can_admit_new_prefetch(self, spec):
        return True  # everything admissible

    def _start_lookup_phase(self, rid, spec):
        self.started.append(rid)


# Build the real patched method by applying the patch to a stub module.
class _StubPC:
    pass


class _Spec:
    def __init__(self, n):
        self.key_groups = [FakeGroup(n)]


sys.modules["_bidaw_stub_pc"] = _StubPC
# The patch imports the real lmcache module; run it inside the venv where
# lmcache is importable (this script is executed with the venv python).
apply_dual_queue_patch()

# Grab the patched methods from the REAL PrefetchController class.
# Third Party (LMCache)
from lmcache.v1.distributed.storage_controllers import (  # noqa: E402
    prefetch_controller as pc,
)

start_method = pc.PrefetchController._start_pending_requests

# Case A: fresh queue, small request after big request -> small first
#   (R = 1 + wait/size; realistic stamps: big enqueued slightly earlier,
#   but /500 vs /5 makes the small request's ratio win immediately)
ctrl = FakeCtrl([(1, _Spec(500)), (2, _Spec(5))])
now = time.monotonic()
ctrl._bidaw_enqueue_monotonic[1] = now - 0.010  # big: arrived 10ms ago
ctrl._bidaw_enqueue_monotonic[2] = now - 0.008  # small: arrived 8ms ago
start_method(ctrl)
check("A: small-before-big (HRRN)", ctrl.started == [2, 1])

# Case B: aging rescues the big request (it waited much longer)
ctrl = FakeCtrl([(1, _Spec(500)), (2, _Spec(5))])
now = time.monotonic()
ctrl._bidaw_enqueue_monotonic[1] = now - 600  # waited 10 min
ctrl._bidaw_enqueue_monotonic[2] = now  # just arrived
start_method(ctrl)
check("B: aging rescues big request", ctrl.started == [1, 2])

# Case C: inadmissible head is skipped, not blocking
ctrl = FakeCtrl([(1, _Spec(500)), (2, _Spec(5))])
ctrl._can_admit_new_prefetch = lambda spec: len(spec.key_groups[0].keys) <= 10
start_method(ctrl)
check("C: head-of-line skipped", ctrl.started == [2])

# Case D: nothing admissible -> no crash, queue preserved
ctrl = FakeCtrl([(1, _Spec(500))])
ctrl._can_admit_new_prefetch = lambda spec: False
start_method(ctrl)
check("D: stall preserved when nothing fits", len(ctrl._pending_queue) == 1)

# Case E: stats recorded
check("E: dispatch counters", STATS.dispatches >= 3 and STATS.reordered_dispatches >= 1)

ok = all(c for _, c in results)
for name, c in results:
    print(("PASS" if c else "FAIL"), name)
print("ALL_OK" if ok else "SOME_FAILED")
sys.exit(0 if ok else 1)
