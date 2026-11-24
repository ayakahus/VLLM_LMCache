# SPDX-License-Identifier: Apache-2.0
"""Log confirmed lifecycle and queue events for LMCache 0.5.7.post1."""
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from uuid import uuid4

from bidaw.padded_tail import TAIL_VALID_TOKENS

_log = None
_logger = logging.getLogger(__name__)
_l1_cycle = ContextVar("bidaw_l1_cycle", default=None)


def l1_cycle_active():
    return _l1_cycle.get() is not None


def record_l1_cycle_phase(field, duration_ms):
    cycle = _l1_cycle.get()
    if cycle is not None:
        cycle[field] = cycle.get(field, 0.0) + duration_ms


@dataclass
class TransferTiming:
    started: int
    round_id: str | None
    dispatcher: object = None
    reserve_ns: int = 0
    reserve_calls: int = 0
    alloc_failed: bool = False
    reported: bool = False


def _install_transfer_timing(log, storage_manager):
    from lmcache.v1.mp_observability.event import EventType
    from lmcache.v1.mp_observability.event_bus import EventBus
    from lmcache.v1.multiprocess.modules.engine_driven_transfer import EngineDrivenTransferModule
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import LMCacheDrivenTransferModule
    from lmcache.v1.multiprocess.native_completion import submit_callback_to_stream

    active = ContextVar("bidaw_mp_transfer_timing", default=None)
    reserve_write = storage_manager.reserve_write
    init = LMCacheDrivenTransferModule.__init__
    publish = EventBus.publish
    publish_on_stream = EventBus.publish_on_stream
    start_types = {EventType.MP_STORE_START, EventType.MP_RETRIEVE_START}
    end_types = {
        EventType.MP_STORE_END: ("store_done", "stored_count"),
        EventType.MP_RETRIEVE_END: ("retrieve_done", "retrieved_count"),
    }
    callback = "bidaw_transfer_timing"

    def emit(kind, fields, *, completed=True):
        fields = dict(fields)
        now = time.monotonic_ns()
        fields["duration_ms"] = (now - fields["started_ns"]) / 1e6 if completed else None
        log(kind, [fields], now)

    def complete(payload):
        kind, fields = payload
        emit(kind, fields, completed=fields["num_keys"] > 0)

    @wraps(init)
    def init_module(self, ctx):
        init(self, ctx)
        self._device_host_func_dispatcher.register(callback, complete, payload_type=tuple[str, dict])

    def timed(native, *, store=False, on_stream=False):
        @wraps(native)
        def call(self, key, *args, **kwargs):
            round_id = (key.request_configs or {}).get("lmcache.bidaw_round_id")
            timing = TransferTiming(time.monotonic_ns(), round_id,
                                    self._device_host_func_dispatcher if on_stream else None)
            token = active.set(timing)
            try:
                return native(self, key, *args, **kwargs)
            finally:
                active.reset(token)
                if store and not timing.reported and (timing.reserve_calls or on_stream):
                    emit("store_done", {
                        "request_id": key.request_id, "round_id": round_id,
                        "started_ns": timing.started, "bytes": 0, "num_keys": 0,
                        "reserve_ms": timing.reserve_ns / 1e6,
                        "alloc_failed": timing.alloc_failed, "completion": "unobserved",
                    }, completed=False)
        return call

    @wraps(reserve_write)
    def reserve(self, keys, layout_desc):
        timing = active.get()
        if timing is None:
            return reserve_write(self, keys, layout_desc)
        started = time.monotonic_ns()
        try:
            return reserve_write(self, keys, layout_desc)
        finally:
            timing.reserve_ns += time.monotonic_ns() - started
            timing.reserve_calls += 1

    def completion_fields(kind, count, event, timing):
        metadata = event.metadata
        fields = {
            "request_id": event.session_id,
            "round_id": timing.round_id if timing and timing.dispatcher else metadata.get("bidaw_round_id"),
            "started_ns": timing.started if timing and timing.dispatcher else metadata["bidaw_started_ns"],
            "bytes": metadata.get("total_bytes", 0),
            "num_keys": metadata.get(count, 0),
            "transfer_key": metadata.get("transfer_key"),
            "completion": "stream_callback" if timing and timing.dispatcher else "engine_commit",
        }
        if kind == "store_done":
            reserve_ns = metadata.get("bidaw_reserve_ns", 0)
            if timing:
                reserve_ns += timing.reserve_ns
            fields.update(reserve_ms=reserve_ns / 1e6,
                          alloc_failed=bool(metadata.get("bidaw_alloc_failed")
                                            or (timing and timing.alloc_failed)))
        return fields

    def stamp(event, timing):
        event.metadata.update(bidaw_started_ns=timing.started, bidaw_round_id=timing.round_id,
                              bidaw_reserve_ns=timing.reserve_ns, bidaw_alloc_failed=int(timing.alloc_failed))

    @wraps(publish)
    def publish_event(self, event):
        timing = active.get()
        if timing is not None and event.event_type == EventType.L1_ALLOCATION_FAILED:
            timing.alloc_failed |= event.metadata.get("during") == "l1_store"
        if timing is not None and event.event_type in start_types and timing.dispatcher is None:
            stamp(event, timing)
            timing.reported = True
        if event.event_type in end_types and "bidaw_started_ns" in event.metadata:
            kind, count = end_types[event.event_type]
            fields = completion_fields(kind, count, event, timing)
            if timing is not None:
                timing.reported = True
            emit(kind, fields, completed=fields["num_keys"] > 0)
        return publish(self, event)

    @wraps(publish_on_stream)
    def stream_event(self, stream, event):
        result = publish_on_stream(self, stream, event)
        timing = active.get()
        if timing is not None and timing.dispatcher and event.event_type in end_types:
            kind, count = end_types[event.event_type]
            fields = completion_fields(kind, count, event, timing)
            timing.reported = True
            try:
                submit_callback_to_stream(stream, callback, (kind, fields))
            except Exception:
                _logger.exception("Could not register Bidaw transfer timing callback")
        return result

    storage_manager.reserve_write = reserve
    EventBus.publish = publish_event
    EventBus.publish_on_stream = stream_event
    LMCacheDrivenTransferModule.__init__ = init_module
    LMCacheDrivenTransferModule.store_with_chunk_mask = timed(
        LMCacheDrivenTransferModule.store_with_chunk_mask, store=True, on_stream=True)
    LMCacheDrivenTransferModule.retrieve = timed(LMCacheDrivenTransferModule.retrieve, on_stream=True)
    EngineDrivenTransferModule.prepare_store = timed(EngineDrivenTransferModule.prepare_store, store=True)
    EngineDrivenTransferModule.commit_store = timed(EngineDrivenTransferModule.commit_store, store=True)
    EngineDrivenTransferModule.prepare_retrieve = timed(EngineDrivenTransferModule.prepare_retrieve)



def record_round(cache_salt, round_index, phase, round_id=None, *, shared_cache=False):
    if _log is None:
        return False
    fields = {"salt": cache_salt, "round": round_index, "round_id": round_id}
    if shared_cache:
        fields["shared_cache"] = True
    _log(f"round_{phase}", [fields])
    return True


def record_shared_replay_end():
    if _log is None:
        return False
    _log("shared_replay_end", [{}])
    return True


def record_preparing_dispatch(controller, request_id):
    if _log is not None:
        _log("preparing", [_request_fields(controller, request_id)])


def record_tier_event(event, key, **fields):
    if _log is not None:
        _log(event, [{**_key_fields(key), **fields}])


def _request_fields(controller, request_id):
    return {"request": f"{id(controller)}:{request_id}"}


def _key_fields(key):
    identity = key.identity
    fields = {
        "model": identity.model_name,
        "rank": identity.kv_rank,
        "group": identity.object_group_id,
        "salt": identity.cache_salt,
        "hash": identity.chunk_hash.hex(),
    }
    round_id = key.extra_attributes.get("lmcache.bidaw_round_id")
    if isinstance(round_id, str) and round_id:
        fields["round_id"] = round_id
    return fields


def apply_l2_event_probe() -> bool:
    global _log
    out = os.environ.get("BIDAW_L2_EVENT_LOG")
    if not out:
        return False
    if _log is not None:
        return True

    from lmcache.v1.distributed.l1_manager.base import BaseL1Manager
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2Adapter
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.distributed.storage_controllers import (
        eviction_controller as ec,
        prefetch_controller as pc,
    )

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8")
    lock = threading.Lock()
    run = uuid4().hex
    load_request = ContextVar("bidaw_load_request", default={})
    load_task = ContextVar("bidaw_load_task", default=None)
    l1_evicting = ContextVar("bidaw_l1_evicting", default=None)
    l2_evicting = ContextVar("bidaw_l2_evicting", default=False)

    def log(event, records, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic_ns()
        lines = [json.dumps({"run": run, "t_ns": timestamp, "ev": event, **fields}) + "\n"
                 for fields in records]
        if lines:
            with lock:
                stream.writelines(lines)
                stream.flush()

    def log_keys(event, keys, extras, timestamp=None):
        log(event, ({**_key_fields(key), **extras} for key in keys), timestamp)

    notify_stored = FSL2Adapter._notify_keys_stored
    notify_accessed = FSL2Adapter._notify_keys_accessed
    execute_load = FSL2Adapter._execute_load
    execute_delete = FSL2Adapter._execute_delete
    submit_prefetch = StorageManager.submit_prefetch_task
    submit_loads = pc.PrefetchController._submit_load_tasks
    publish_result = pc.PrefetchController._publish_result
    evict_l1 = ec.L1EvictionController.execute_eviction_action
    evict_l2 = ec.L2EvictionController._execute_eviction_action
    init_l1 = ec.L1EvictionController.__init__
    l1_deleted = BaseL1Manager._free_and_report_deleted

    def stored(self, keys, sizes):
        def write_records():
            for key, size in zip(keys, sizes, strict=True):
                fields = {**_key_fields(key), "adapter": str(self._base_path), "bytes": size}
                valid = key.extra_attributes.get(TAIL_VALID_TOKENS)
                if valid is not None:
                    fields["tail_valid_tokens"] = valid
                yield fields

        log("write", write_records())
        return notify_stored(self, keys, sizes)

    def load_fields(adapter):
        fields = {"adapter": str(adapter._base_path), **load_request.get()}
        task = load_task.get()
        if task is not None:
            fields["io_task"] = task
        return fields

    def accessed(self, keys):
        # submit_touch also sends this notification, without reading any bytes.
        if load_task.get() is not None:
            log_keys("load", keys, load_fields(self))
        return notify_accessed(self, keys)

    async def load(self, keys, objects, task_id):
        token = load_task.set(f"{id(self)}:{task_id}")
        try:
            log_keys("load_attempt", keys, load_fields(self))
            return await execute_load(self, keys, objects, task_id)
        finally:
            load_task.reset(token)

    async def delete(self, keys):
        removed_keys, sizes = await execute_delete(self, keys)
        log_keys("delete", removed_keys, {
            "adapter": str(self._base_path), "eviction": l2_evicting.get(),
        })
        return removed_keys, sizes

    @wraps(init_l1)
    def init_l1_controller(self, *args, **kwargs):
        init_l1(self, *args, **kwargs)
        if os.environ.get("GANDALF_TIER_MODE") != "lru_writeback":
            return
        select = self._eviction_policy.get_eviction_actions

        @wraps(select)
        def timed_selection(*args, **kwargs):
            started = time.monotonic_ns()
            actions = select(*args, **kwargs)
            self._bidaw_native_selection_ms = (time.monotonic_ns() - started) / 1e6
            return actions

        self._eviction_policy.get_eviction_actions = timed_selection

    def l1_action(self, action):
        policy = self._eviction_policy.name
        bounded = getattr(self, "_gandalf_budget", None)
        if bounded is not None and bounded.settings.mode != "write_through":
            policy = bounded.settings.mode
        cycle = {"policy": policy, "requested_keys": len(action.keys),
                 "evicted_keys": 0, "evicted_bytes": 0}
        if policy in {"gandalf", "lru_writeback"}:
            cycle.update(selection_ms=0.0, ssd_write_ms=0.0)
        if policy == "lru_writeback":
            cycle["selection_ms"] = getattr(self, "_bidaw_native_selection_ms", None)
        cycle_token = _l1_cycle.set(cycle)
        token = l1_evicting.set((self._l1_manager, policy))
        started = time.monotonic_ns()
        try:
            return evict_l1(self, action)
        finally:
            finished = time.monotonic_ns()
            l1_evicting.reset(token)
            _l1_cycle.reset(cycle_token)
            cycle["duration_ms"] = (finished - started) / 1e6
            log("l1_evict_cycle", [cycle], finished)

    def l1_removed(self, keys, objs):
        context = l1_evicting.get()
        same_manager = context is not None and context[0] is self
        evicted_bytes = sum(obj.get_size() for obj in objs) if same_manager else 0
        l1_deleted(self, keys, objs)
        # The manager reports expired staging reservations via a separate path.
        if same_manager:
            cycle = _l1_cycle.get()
            cycle["evicted_keys"] += len(keys)
            cycle["evicted_bytes"] += evicted_bytes
            log_keys("l1_evict", keys, {"policy": context[1]})
        elif keys:
            log_keys("l1_remove", keys, {})

    def l2_action(self, adapter, action):
        token = l2_evicting.set(True)
        try:
            return evict_l2(self, adapter, action)
        finally:
            l2_evicting.reset(token)

    def prefetch(self, spec, external_request_id=""):
        started = time.monotonic_ns()
        handle = submit_prefetch(self, spec, external_request_id)
        # Fast-lane results can publish before submit returns. Preserve the
        # submission time so the report can correlate them in causal order.
        fields = _request_fields(self._prefetch_controller, handle.prefetch_request_id)
        log_keys("lookup", (key for group in spec.key_groups for key in group.keys),
                 fields, started)
        return handle

    def schedule_loads(self, request, reserved_objs):
        # run_coroutine_threadsafe propagates this context to the FS loop.
        token = load_request.set(_request_fields(self, request.request_id))
        try:
            return submit_loads(self, request, reserved_objs)
        finally:
            load_request.reset(token)

    def publish(self, request, result):
        fields = _request_fields(self, request.request_id)
        if result.l1_hit_cells:
            log_keys("l1_hit", pc._gather_keys(request.key_groups, result.l1_hit_cells), fields)
        if result.l2_hit_cells:
            log_keys("l2_hit", pc._gather_keys(request.key_groups, result.l2_hit_cells), fields)
        return publish_result(self, request, result)

    log("probe_on", [{"version": 4}])
    log("timing_probe_on", [{"version": 1}])
    FSL2Adapter._notify_keys_stored = stored
    FSL2Adapter._notify_keys_accessed = accessed
    FSL2Adapter._execute_load = load
    FSL2Adapter._execute_delete = delete
    StorageManager.submit_prefetch_task = prefetch
    pc.PrefetchController._submit_load_tasks = schedule_loads
    pc.PrefetchController._publish_result = publish
    ec.L1EvictionController.__init__ = init_l1_controller
    ec.L1EvictionController.execute_eviction_action = l1_action
    ec.L2EvictionController._execute_eviction_action = l2_action
    BaseL1Manager._free_and_report_deleted = l1_removed
    _install_transfer_timing(log, StorageManager)
    _log = log
    return True
