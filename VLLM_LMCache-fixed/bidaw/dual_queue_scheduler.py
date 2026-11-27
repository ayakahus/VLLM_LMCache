# SPDX-License-Identifier: Apache-2.0
"""LMCache 0.5.7.post1 prefetch fast lane and disk-HRRN SSD dispatch.

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
        self.last_dispatch_bytes = 0

    def snapshot(self):
        with self._lock:
            return dict(enabled=self.enabled, fast_lane_hits=self.fast_lane_hits,
                        fast_lane_fallbacks=self.fast_lane_fallbacks,
                        dispatches=self.dispatches,
                        reordered_dispatches=self.reordered_dispatches,
                        max_pending_wait_s=round(self.max_wait_s, 6),
                        last_dispatch_ratio=self.last_dispatch_ratio,
                        last_dispatch_bytes=self.last_dispatch_bytes)


STATS = DualQueueStats()


def scheduler_snapshot():
    return STATS.snapshot()


def apply_dual_queue_patch():
    from lmcache.v1.distributed.storage_controllers import prefetch_controller as pc
    from bidaw import policy_registry
    from bidaw.l2_event_probe import record_preparing_dispatch

    cls = pc.PrefetchController
    if getattr(cls, '_bidaw_access_patch', False):
        return STATS.enabled
    enabled = os.environ.get('BIDAW_DUAL_QUEUE', '1') != '0'
    disk_limit = 4
    if enabled:
        try:
            disk_limit = int(os.environ.get('BIDAW_L2_READ_CONCURRENCY', '16'))
        except ValueError as exc:
            raise ValueError('BIDAW_L2_READ_CONCURRENCY must be a positive integer') from exc
        if disk_limit < 1:
            raise ValueError('BIDAW_L2_READ_CONCURRENCY must be a positive integer')
    STATS.enabled = enabled
    original_init = cls.__init__
    original_submit = cls.submit_prefetch_request
    original_advance = cls._advance_request
    original_retire = cls._retire_request
    original_cleanup = cls._cleanup_in_flight_requests

    def init(self, *args, **kwargs):
        # Must exist before the controller can start its event-loop thread.
        self._bidaw_enqueue_monotonic = {}
        self._bidaw_waiting_locks = {}
        self._bidaw_disk_ready = {}
        self._bidaw_disk_in_flight = set()
        self._bidaw_dispatching = False
        original_init(self, *args, **kwargs)

    def submit(self, spec):
        # Salted sessions are observed at the MP lookup handler, even for
        # requests with no complete chunk. Shared keys account for traffic.
        shared_objects = {}
        for group in spec.key_groups:
            shared_keys = [key for key in group.keys if not key.cache_salt]
            if shared_keys:
                layout = group.layout_desc
                size = pc.get_size_bytes(layout.shapes, layout.dtypes)
                shared_objects.update(dict.fromkeys(shared_keys, size))
        policy_registry.feed_access(shared_objects)
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
        try:
            # Timestamp and locks become visible before the event-loop wakeup.
            with self._submission_lock:
                if self._stop_flag.is_set():
                    raise RuntimeError('Prefetch controller is stopping')
                self._bidaw_enqueue_monotonic[rid] = enqueued_at
                self._bidaw_waiting_locks[rid] = request
                self._submission_queue.append((rid, spec))
                try:
                    self._submission_efd.notify()
                except Exception:
                    self._submission_queue.pop()
                    del self._bidaw_waiting_locks[rid]
                    del self._bidaw_enqueue_monotonic[rid]
                    raise
        except Exception:
            self._release_all_locks(request)
            raise
        with STATS._lock:
            STATS.fast_lane_fallbacks += 1
        return rid

    def start_pending(self):
        dispatch_disk(self)
        pending = self._pending_queue
        while pending:
            index = next((i for i, (_, spec) in enumerate(pending)
                          if self._can_admit_new_prefetch(spec)), None)
            if index is None:
                return
            rid, spec = pending[index]
            del pending[index]
            with self._submission_lock:
                waiting = self._bidaw_waiting_locks.pop(rid)
            try:
                self._start_lookup_phase(rid, spec)
            except Exception:
                pc.logger.exception('Failed starting prefetch request %d', rid)
                request = self._in_flight_requests.get(rid)
                if request is not None:
                    self._abort_request(request)
            finally:
                # The native start has acquired its own reader shares. Keep
                # the original ones until then to close the eviction gap.
                self._release_all_locks(waiting)
                if rid not in self._in_flight_requests:
                    with self._submission_lock:
                        self._bidaw_enqueue_monotonic.pop(rid, None)

    def stage_disk(self, request):
        active_l2_descs = {
            index: desc for index, desc in self._adapter_descriptors.items()
            if self._is_l2_eligible(request, index)
        }
        # Keep only the native plan's read locks while waiting for a disk slot.
        # Native _plan_load only prunes held locks. The transition revalidates
        # those locks before allocation without acquiring them a second time.
        self._plan_load(request, active_l2_descs)
        row_sizes = [
            pc.get_size_bytes(group.layout_desc.shapes, group.layout_desc.dtypes)
            for group in request.key_groups
        ]
        size_bytes = sum(
            row.popcount() * row_size
            for _, cells in request.key_states.l2_locked_keys.items()
            for row, row_size in zip(cells, row_sizes, strict=True)
        )
        if size_bytes:
            self._bidaw_disk_ready[request.request_id] = request, size_bytes
            dispatch_disk(self)
        else:
            self._finish_request(request)
            self._retire_request(request)

    def advance(self, request, signaled_adapters):
        if request.request_id in self._bidaw_disk_ready:
            return
        if request.phase != pc.PrefetchPhase.LOOKUP:
            return original_advance(self, request, signaled_adapters)
        adapters = signaled_adapters[pc.PrefetchPhase.LOOKUP]
        if not adapters:
            return
        self._poll_lookup_results(request, adapters)
        if request.all_lookups_done():
            stage_disk(self, request)

    def dispatch_disk(self):
        ready = self._bidaw_disk_ready
        in_flight = self._bidaw_disk_in_flight
        if not ready or len(in_flight) >= disk_limit:
            return
        if self._bidaw_dispatching or self._stop_flag.is_set():
            return
        self._bidaw_dispatching = True
        try:
            now = time.monotonic()
            with self._submission_lock:
                arrival = {rid: self._bidaw_enqueue_monotonic[rid] for rid in ready}
            ordered = sorted(ready, key=lambda rid: (
                -(now - arrival[rid]) / ready[rid][1], arrival[rid], rid
            ))
            for rid in ordered:
                if len(in_flight) >= disk_limit:
                    break
                reordered = rid != next(iter(ready))
                request, size_bytes = ready.pop(rid)
                wait = now - arrival[rid]
                in_flight.add(rid)
                try:
                    record_preparing_dispatch(self, rid)
                    self._transition_to_load_phase(request)
                    if request.all_loads_done():
                        self._finish_request(request)
                        self._retire_request(request)
                except Exception:
                    pc.logger.exception('Failed dispatching prefetch request %d', rid)
                    self._abort_request(request)
                    continue
                with STATS._lock:
                    STATS.dispatches += 1
                    STATS.reordered_dispatches += int(reordered)
                    STATS.max_wait_s = max(STATS.max_wait_s, wait)
                    STATS.last_dispatch_ratio = 1.0 + wait / size_bytes
                    STATS.last_dispatch_bytes = size_bytes
        finally:
            self._bidaw_dispatching = False

    def retire(self, request):
        rid = request.request_id
        self._bidaw_disk_ready.pop(rid, None)
        released = rid in self._bidaw_disk_in_flight
        self._bidaw_disk_in_flight.discard(rid)
        with self._submission_lock:
            self._bidaw_enqueue_monotonic.pop(rid, None)
        result = original_retire(self, request)
        if released:
            dispatch_disk(self)
        return result

    def cleanup(self):
        try:
            return original_cleanup(self)
        finally:
            with self._submission_lock:
                waiting = list(self._bidaw_waiting_locks.values())
                self._bidaw_waiting_locks.clear()
                self._bidaw_enqueue_monotonic.clear()
            self._bidaw_disk_ready.clear()
            self._bidaw_disk_in_flight.clear()
            for request in waiting:
                self._release_all_locks(request)

    cls.__init__ = init
    cls.submit_prefetch_request = submit
    if enabled:
        cls._start_pending_requests = start_pending
        cls._advance_request = advance
        cls._retire_request = retire
        cls._cleanup_in_flight_requests = cleanup
    cls._bidaw_access_patch = True
    pc.logger.info('Bidaw logical access observer enabled; dual_queue=%s', enabled)
    return enabled
