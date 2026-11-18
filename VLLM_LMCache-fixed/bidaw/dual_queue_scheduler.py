# SPDX-License-Identifier: Apache-2.0
"""LMCache 0.5.7.post1 prefetch fast lane and disk-HRRN admission.

This is a storage-side scheduler, not a replacement for vLLM's GPU scheduler.
Logical access observation is installed even when BIDAW_DUAL_QUEUE=0.
"""
import os
import threading
import time


class DualQueueStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.enabled = False
        self.fast_lane_hits = 0
        self.fast_lane_fallbacks = 0
        self.reordered_dispatches = 0
        self.dispatches = 0
        self.max_wait_s = 0.0
        self.last_dispatch_ratio = 0.0

    def snapshot(self):
        with self._lock:
            return dict(enabled=self.enabled, fast_lane_hits=self.fast_lane_hits,
                        fast_lane_fallbacks=self.fast_lane_fallbacks,
                        dispatches=self.dispatches,
                        reordered_dispatches=self.reordered_dispatches,
                        max_pending_wait_s=round(self.max_wait_s, 6),
                        last_dispatch_ratio=self.last_dispatch_ratio)


STATS = DualQueueStats()


def scheduler_snapshot():
    return STATS.snapshot()


def _num_keys(spec):
    return max(1, sum(len(group.keys) for group in spec.key_groups))


def apply_dual_queue_patch():
    from lmcache.v1.distributed.storage_controllers import prefetch_controller as pc
    from bidaw import policy_registry

    cls = pc.PrefetchController
    if getattr(cls, '_bidaw_access_patch', False):
        return STATS.enabled
    enabled = os.environ.get('BIDAW_DUAL_QUEUE', '1') != '0'
    STATS.enabled = enabled
    original_init = cls.__init__
    original_submit = cls.submit_prefetch_request

    def init(self, *args, **kwargs):
        # Must exist before the controller can start its event-loop thread.
        self._bidaw_enqueue_monotonic = {}
        original_init(self, *args, **kwargs)

    def submit(self, spec):
        # Exactly once per submitted spec; L1 lifecycle callbacks do not train.
        policy_registry.feed_access([k for g in spec.key_groups for k in g.keys])
        if not enabled or not spec.key_groups:
            return original_submit(self, spec)
        enqueued_at = time.monotonic()
        with self._submission_lock:
            rid = self._next_request_id
            self._next_request_id += 1
        request = pc._build_request(rid, spec)
        try:
            self._lock_l1_keys(request)
            needs_l2 = (bool(self._l2_adapters)
                        and spec.allowed_l2_memory_tags != frozenset()
                        and pc._needs_l2_lookup(request.key_groups,
                            request.key_states.l1_locked_keys,
                            request.fetching_policy))
            if not needs_l2:
                self._plan_load(request, {})
                self._finish_request(request)
                with STATS._lock:
                    STATS.fast_lane_hits += 1
                return rid
        except Exception:
            self._release_all_locks(request)
            raise
        self._release_all_locks(request)
        # Timestamp and request become visible atomically, before wakeup.
        with self._submission_lock:
            self._bidaw_enqueue_monotonic[rid] = enqueued_at
            self._submission_queue.append((rid, spec))
            self._submission_efd.notify()
        with STATS._lock:
            STATS.fast_lane_fallbacks += 1
        return rid

    def start_pending(self):
        pending = self._pending_queue
        while pending:
            now = time.monotonic()
            best_i, best_r = None, -1.0
            for i, (rid, spec) in enumerate(pending):
                if not self._can_admit_new_prefetch(spec):
                    continue
                with self._submission_lock:
                    enqueued = self._bidaw_enqueue_monotonic.get(rid, now)
                ratio = 1.0 + max(0, now - enqueued) / _num_keys(spec)
                if ratio > best_r:
                    best_i, best_r = i, ratio
            if best_i is None:
                return
            rid, spec = pending[best_i]
            del pending[best_i]
            with self._submission_lock:
                enqueued = self._bidaw_enqueue_monotonic.pop(rid, now)
            with STATS._lock:
                STATS.dispatches += 1
                STATS.reordered_dispatches += int(best_i != 0)
                STATS.max_wait_s = max(STATS.max_wait_s, now - enqueued)
                STATS.last_dispatch_ratio = best_r
            try:
                self._start_lookup_phase(rid, spec)
            except Exception:
                pc.logger.exception('Failed starting prefetch request %d', rid)
                request = self._in_flight_requests.get(rid)
                if request is not None:
                    self._abort_request(request)

    cls.__init__ = init
    cls.submit_prefetch_request = submit
    if enabled:
        cls._start_pending_requests = start_pending
    cls._bidaw_access_patch = True
    pc.logger.info('Bidaw logical access observer enabled; dual_queue=%s', enabled)
    return enabled
