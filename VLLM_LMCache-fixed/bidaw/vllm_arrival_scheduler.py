# SPDX-License-Identifier: Apache-2.0
"""FCFS GPU admission across vLLM's two waiting queues."""
import inspect
import json
import logging
import os
import time
from importlib.metadata import version
from textwrap import dedent
from urllib.request import urlopen

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler

from bidaw.async_jsonl import TimingLog

logger = logging.getLogger(__name__)

# vLLM 0.33.0 has no override point for candidate selection. Keep the
# allocation and request state transitions in the installed scheduler.
_NATIVE_QUEUE_SELECTION = (
    "                # Requests holding KV blocks are always drained first, and are\n"
    "                # the only ones admitted under PAUSED_NEW (e.g. async KV loads).\n"
    "                request_queue = self.kv_holding_waiting\n"
    "                if not request_queue and self._pause_state == PauseState.UNPAUSED:\n"
    "                    request_queue = self.waiting\n"
)
_ARRIVAL_QUEUE_SELECTION = (
    "                # Try the oldest eligible waiting request first.\n"
    "                request_queue = _bidaw_next_queue(self)\n"
)
_NATIVE_ALLOCATION_FAILURE = (
    "                    if request.has_encoder_inputs:\n"
    "                        self.encoder_cache_manager.free(request)\n"
    "                    break\n"
    "\n"
    "                # KVTransfer: the connector uses this info to determine\n"
)
_SKIP_ALLOCATION_FAILURE = _NATIVE_ALLOCATION_FAILURE.replace(
    "                    break\n",
    "                    skip_request(request_queue)\n"
    "                    # This request is ready but still needs memory.\n"
    "                    self.deferred_waiting.discard(request)\n"
    "                    continue\n",
)


def _arrival_key(request):
    return request.arrival_time, request.request_id


def _next_queue(scheduler):
    holding = scheduler.kv_holding_waiting
    waiting = scheduler.waiting
    if scheduler._pause_state != PauseState.UNPAUSED or not waiting:
        return holding
    if not holding or (
        _arrival_key(waiting.peek_request()) < _arrival_key(holding.peek_request())
    ):
        return waiting
    return holding


def _compile_arrival_schedule(native, *, dual_queue=True, timing=False):
    lines, first_line = inspect.getsourcelines(native)
    source = "".join(lines)
    replacements = []
    if dual_queue:
        replacements.extend((
            (_NATIVE_QUEUE_SELECTION, _ARRIVAL_QUEUE_SELECTION),
            (_NATIVE_ALLOCATION_FAILURE, _SKIP_ALLOCATION_FAILURE),
        ))
    if timing:
        lookup_start = "                        ext_tokens, load_kv_async = (\n"
        lookup_done = "                        if ext_tokens is None:\n"
        load_start = "                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS\n"
        running = "                self.running.append(request)\n"
        replacements.extend((
            (lookup_start, "                        self._bidaw_lookup_start(request)\n"
             + lookup_start),
            (lookup_done, "                        if ext_tokens is not None:\n"
             "                            self._bidaw_stage(request, 't_lookup_done')\n"
             + lookup_done),
            (load_start, "                    self._bidaw_stage(request, 't_kv_load_start')\n"
             + load_start),
            (running, running + "                self._bidaw_running(request, num_computed_tokens)\n"),
        ))
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise RuntimeError("The Bidaw arrival scheduler requires the vLLM 0.33.0 scheduling loop")
        source = source.replace(original, replacement, 1)
    namespace = dict(native.__globals__, _bidaw_next_queue=_next_queue)
    code = compile("\n" * (first_line - 1) + dedent(source), native.__code__.co_filename, "exec")
    exec(code, namespace)
    return namespace[native.__name__]


class ArrivalQueue(FCFSRequestQueue):
    """Track insertions that can change FCFS order; removals preserve ordering."""

    def __init__(self, requests=()):
        super().__init__(requests)
        self._dirty = bool(self)

    def append(self, request):
        if self and _arrival_key(request) < _arrival_key(self[-1]):
            self._dirty = True
        super().append(request)

    def appendleft(self, request):
        if self and _arrival_key(request) > _arrival_key(self[0]):
            self._dirty = True
        super().appendleft(request)

    def extend(self, requests):
        for request in requests:
            self.append(request)

    def extendleft(self, requests):
        super().extendleft(requests)
        self._dirty = len(self) > 1

    def sort_if_needed(self):
        if self._dirty and len(self) > 1:
            ordered = sorted(self, key=_arrival_key)
            self.clear()
            super().extend(ordered)
        self._dirty = False


class BidawScheduler(Scheduler):
    """Select the native sync or async scheduler and enable arrival order for DQ."""

    def __new__(cls, vllm_config, *args, **kwargs):
        if cls is BidawScheduler and vllm_config.scheduler_config.async_scheduling:
            return super().__new__(_AsyncBidawScheduler)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arrival_schedule = None
        path = os.environ.get("BIDAW_VLLM_TIMING_LOG")
        self._timing_log = TimingLog(path) if path else None
        self._request_timing = {}
        self._dual_queue = False
        if self.policy.value == "fcfs" and self.connector is not None:
            dual_queue = getattr(self.connector, "bidaw_dual_queue", None)
            if dual_queue is None:
                # The stock MP connector is used when padded tails are disabled.
                try:
                    port = os.environ.get("BIDAW_LMCACHE_HTTP_PORT", "8080")
                    with urlopen(f"http://127.0.0.1:{port}/bidaw/status", timeout=5) as response:
                        dual_queue = json.load(response).get("scheduler", {}).get("enabled")
                except (OSError, ValueError):
                    logger.warning("Bidaw scheduler status unavailable. Using native vLLM scheduling.")
            self._dual_queue = dual_queue is True
        if self._dual_queue or self._timing_log:
            if version("vllm") != "0.33.0":
                raise RuntimeError("Bidaw scheduling and timing require vLLM 0.33.0")
            self._arrival_schedule = _compile_arrival_schedule(
                Scheduler.schedule, dual_queue=self._dual_queue, timing=bool(self._timing_log))
        if self._dual_queue:
            self.waiting = ArrivalQueue(self.waiting)
            self.kv_holding_waiting = ArrivalQueue(self.kv_holding_waiting)
            logger.info("Bidaw GPU admission uses original request arrival order")

    def _enqueue_waiting_request(self, request):
        super()._enqueue_waiting_request(request)
        if self._timing_log and request.request_id not in self.requests:
            self._request_timing[request.request_id] = {
                "ev": "request_timing", "request_id": request.request_id,
                "round_id": (request.kv_transfer_params or {}).get("lmcache.bidaw_round_id"),
                "num_prompt_tokens": request.num_prompt_tokens, "num_cached_tokens": None,
                "t_arrive": time.monotonic_ns(), "t_lookup_start": None, "t_lookup_done": None,
                "t_kv_load_start": None, "t_kv_load_done": None, "t_running": None,
            }

    def _bidaw_stage(self, request, field, timestamp=None):
        record = self._request_timing.get(request.request_id)
        if record is not None and record[field] is None:
            record[field] = time.monotonic_ns() if timestamp is None else timestamp

    def _bidaw_lookup_start(self, request):
        # Eager LMCache lookup can start on arrival, before the scheduler polls.
        tracker = getattr(self.connector, "request_trackers", {}).get(request.request_id)
        started = getattr(tracker, "lookup_started_at", None)
        self._bidaw_stage(request, "t_lookup_start",
                          None if started is None else int(started * 1e9))

    def _bidaw_running(self, request, cached):
        self._bidaw_stage(request, "t_running")
        record = self._request_timing.pop(request.request_id, None)
        if record is not None:
            record["num_cached_tokens"] = min(cached, request.num_prompt_tokens)
            self._timing_log.write(record)

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        if self._timing_log:
            now = time.monotonic_ns()
            for request_id in kv_connector_output.finished_recving or ():
                request = self.requests.get(request_id)
                if request is not None:
                    self._bidaw_stage(request, "t_kv_load_done", now)
        return super()._update_from_kv_xfer_finished(kv_connector_output)

    def _free_request(self, request, delay_free_blocks=False):
        if self._timing_log:
            record = self._request_timing.pop(request.request_id, None)
            if record is not None:
                self._timing_log.write(record)
        return super()._free_request(request, delay_free_blocks)

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            if self._timing_log:
                self._timing_log.close()
                self._timing_log = None

    def schedule(self, throttle_prefills=False):
        if self._arrival_schedule is None:
            return super().schedule(throttle_prefills)

        # Native skip and promotion paths prepend to their respective queues.
        # Sort both again before selecting across their heads this step.
        if self._dual_queue:
            self.waiting.sort_if_needed()
            self.kv_holding_waiting.sort_if_needed()
        return self._arrival_schedule(self, throttle_prefills)


class _AsyncBidawScheduler(BidawScheduler, AsyncScheduler):
    pass
