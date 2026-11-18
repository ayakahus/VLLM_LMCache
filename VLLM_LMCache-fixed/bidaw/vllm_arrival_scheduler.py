# SPDX-License-Identifier: Apache-2.0
"""FCFS GPU admission across vLLM's two waiting queues."""
import inspect
import json
import logging
from importlib.metadata import version
from textwrap import dedent
from urllib.request import urlopen

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler

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


def _compile_arrival_schedule(native):
    lines, first_line = inspect.getsourcelines(native)
    source = "".join(lines)
    for original, replacement in (
        (_NATIVE_QUEUE_SELECTION, _ARRIVAL_QUEUE_SELECTION),
        (_NATIVE_ALLOCATION_FAILURE, _SKIP_ALLOCATION_FAILURE),
    ):
        if source.count(original) != 1:
            raise RuntimeError("The Bidaw arrival scheduler requires the vLLM 0.33.0 scheduling loop")
        source = source.replace(original, replacement, 1)
    namespace = dict(native.__globals__, _bidaw_next_queue=_next_queue)
    code = compile("\n" * (first_line - 1) + dedent(source), native.__code__.co_filename, "exec")
    exec(code, namespace)
    return namespace[native.__name__]


class BidawScheduler(Scheduler):
    """Select the native sync or async scheduler and enable arrival order for DQ."""

    def __new__(cls, vllm_config, *args, **kwargs):
        if cls is BidawScheduler and vllm_config.scheduler_config.async_scheduling:
            return super().__new__(_AsyncBidawScheduler)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arrival_schedule = None
        if self.policy.value != "fcfs" or self.connector is None:
            return
        dual_queue = getattr(self.connector, "bidaw_dual_queue", None)
        if dual_queue is None:
            # The stock MP connector is used when padded tails are disabled.
            try:
                with urlopen("http://127.0.0.1:8080/bidaw/status", timeout=5) as response:
                    dual_queue = json.load(response).get("scheduler", {}).get("enabled")
            except (OSError, ValueError):
                logger.warning("Bidaw scheduler status unavailable. Using native vLLM scheduling.")
        if dual_queue is True:
            if version("vllm") != "0.33.0":
                raise RuntimeError("The Bidaw arrival scheduler requires vLLM 0.33.0")
            self._arrival_schedule = _compile_arrival_schedule(Scheduler.schedule)
            logger.info("Bidaw GPU admission uses original request arrival order")

    def schedule(self, throttle_prefills=False):
        if self._arrival_schedule is None:
            return super().schedule(throttle_prefills)

        # Native skip and promotion paths prepend to their respective queues.
        # Sort both again before selecting across their heads this step.
        for queue in (self.waiting, self.kv_holding_waiting):
            if len(queue) > 1:
                ordered = sorted(queue, key=_arrival_key)
                queue.clear()
                queue.extend(ordered)
        return self._arrival_schedule(self, throttle_prefills)


class _AsyncBidawScheduler(BidawScheduler, AsyncScheduler):
    pass
