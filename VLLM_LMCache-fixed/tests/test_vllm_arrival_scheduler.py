"""CPU regressions for the pinned vLLM waiting-loop patch."""
import importlib
import io
import json
import sys
import unittest
from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class PauseState(Enum):
    UNPAUSED = 0
    PAUSED_NEW = 1


class Queue(deque):
    def peek_request(self):
        return self[0]


class Scheduler:
    def __init__(self, vllm_config, connector):
        self.connector = connector
        self.policy = SimpleNamespace(value="fcfs")
        self.waiting = Queue()
        self.kv_holding_waiting = Queue()
        self.deferred_waiting = set()
        self._pause_state = PauseState.UNPAUSED
        self.free_blocks = 10
        self.encoder_cache_manager = SimpleNamespace(free=lambda request: None)

    def schedule(self, throttle_prefills=False):
        admitted, skipped = [], []

        def skip_request(queue):
            request = queue.popleft()
            skipped.append(request)
            self.deferred_waiting.add(request)

        if self.waiting or self.kv_holding_waiting:
            while True:
                # Requests holding KV blocks are always drained first, and are
                # the only ones admitted under PAUSED_NEW (e.g. async KV loads).
                request_queue = self.kv_holding_waiting
                if not request_queue and self._pause_state == PauseState.UNPAUSED:
                    request_queue = self.waiting
                if not request_queue:
                    break
                request = request_queue.peek_request()
                if request.preparing:
                    skip_request(request_queue)
                    continue
                new_blocks = request.blocks if self.free_blocks >= request.blocks else None
                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                self.free_blocks -= new_blocks
                self.deferred_waiting.discard(request)
                admitted.append(request_queue.popleft().request_id)

        for request in reversed(skipped):
            queue = self.kv_holding_waiting if request.holder else self.waiting
            queue.appendleft(request)
        self.after_schedule()
        return admitted

    def after_schedule(self):
        pass


class AsyncScheduler(Scheduler):
    def after_schedule(self):
        self.async_step = True


@dataclass(eq=False)
class Request:
    request_id: str
    arrival_time: float
    blocks: int = 1
    preparing: bool = False
    holder: bool = False
    has_encoder_inputs: bool = False


class ArrivalSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = {}
        for module_name, member in (
            ("scheduler", Scheduler), ("async_scheduler", AsyncScheduler), ("interface", PauseState),
        ):
            name = f"vllm.v1.core.sched.{module_name}"
            modules[name] = ModuleType(name)
            setattr(modules[name], member.__name__, member)
        with patch.dict(sys.modules, modules):
            cls.mod = importlib.import_module("bidaw.vllm_arrival_scheduler")

    def make_scheduler(self, dual_queue=True, asynchronous=False):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=asynchronous)
        )
        with patch.object(self.mod, "version", return_value="0.33.0"):
            return self.mod.BidawScheduler(
                vllm_config=config, connector=SimpleNamespace(bidaw_dual_queue=dual_queue)
            )

    def test_promoted_request_keeps_original_arrival_across_queues(self):
        scheduler = self.make_scheduler()
        first = Request("first", 1, preparing=True)
        scheduler.waiting.extend([Request("later", 3), first])
        self.assertEqual(scheduler.schedule(), ["later"])
        first.preparing = False
        scheduler.kv_holding_waiting.append(Request("holder", 2, holder=True))
        self.assertEqual(scheduler.schedule(), ["first", "holder"])

    def test_requeued_kv_holders_are_sorted_with_normal_requests(self):
        scheduler = self.make_scheduler(asynchronous=True)
        self.assertIsInstance(scheduler, AsyncScheduler)
        scheduler.kv_holding_waiting.extend([
            Request("third", 3, holder=True), Request("first", 1, holder=True),
        ])
        scheduler.waiting.append(Request("second", 2))
        self.assertEqual(scheduler.schedule(), ["first", "second", "third"])
        self.assertTrue(scheduler.async_step)

    def test_allocation_failure_skips_head_through_native_requeue(self):
        scheduler = self.make_scheduler()
        scheduler.free_blocks = 4
        blocked = Request("too large", 1, blocks=6)
        scheduler.waiting.append(blocked)
        scheduler.kv_holding_waiting.append(Request("fits", 2, blocks=2, holder=True))
        self.assertEqual(scheduler.schedule(), ["fits"])
        self.assertEqual(list(scheduler.waiting), [blocked])
        self.assertNotIn(blocked, scheduler.deferred_waiting)
        scheduler.free_blocks = 6
        self.assertEqual(scheduler.schedule(), ["too large"])

    def test_pause_new_still_drains_only_kv_holders(self):
        scheduler = self.make_scheduler()
        scheduler._pause_state = PauseState.PAUSED_NEW
        scheduler.waiting.append(Request("first", 1))
        scheduler.kv_holding_waiting.append(Request("holder", 2, holder=True))
        self.assertEqual(scheduler.schedule(), ["holder"])

    def test_dual_queue_off_keeps_native_order(self):
        scheduler = self.make_scheduler(dual_queue=False)
        scheduler.waiting.append(Request("first", 1))
        scheduler.kv_holding_waiting.append(Request("holder", 2, holder=True))
        self.assertEqual(scheduler.schedule(), ["holder", "first"])

    def test_stock_connector_reads_server_dual_queue_status(self):
        config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=False))
        status = {"scheduler": {"enabled": True}}
        with (
            patch.object(self.mod, "version", return_value="0.33.0"),
            patch.object(self.mod, "urlopen", return_value=io.StringIO(json.dumps(status))),
        ):
            scheduler = self.mod.BidawScheduler(config, connector=object())
        self.assertIsNotNone(scheduler._arrival_schedule)
