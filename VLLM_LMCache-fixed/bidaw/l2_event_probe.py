# SPDX-License-Identifier: Apache-2.0
"""Log confirmed lifecycle and queue events for LMCache 0.5.7.post1."""
import json
import os
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from bidaw.padded_tail import TAIL_VALID_TOKENS

_log = None


def record_round(cache_salt, round_index, phase, round_id=None):
    if _log is None:
        return False
    _log(f"round_{phase}", [{"salt": cache_salt, "round": round_index, "round_id": round_id}])
    return True


def record_preparing_dispatch(controller, request_id):
    if _log is not None:
        _log("preparing", [_request_fields(controller, request_id)])


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
        with lock:
            if timestamp is None:
                timestamp = time.monotonic_ns()
            for fields in records:
                record = {"run": run, "t_ns": timestamp, "ev": event, **fields}
                stream.write(json.dumps(record) + "\n")
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

    def l1_action(self, action):
        token = l1_evicting.set((self._l1_manager, self._eviction_policy.name))
        try:
            return evict_l1(self, action)
        finally:
            l1_evicting.reset(token)

    def l1_removed(self, keys, objs):
        l1_deleted(self, keys, objs)
        context = l1_evicting.get()
        # The manager reports expired staging reservations via a separate path.
        if context is not None and context[0] is self:
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
    FSL2Adapter._notify_keys_stored = stored
    FSL2Adapter._notify_keys_accessed = accessed
    FSL2Adapter._execute_load = load
    FSL2Adapter._execute_delete = delete
    StorageManager.submit_prefetch_task = prefetch
    pc.PrefetchController._submit_load_tasks = schedule_loads
    pc.PrefetchController._publish_result = publish
    ec.L1EvictionController.execute_eviction_action = l1_action
    ec.L2EvictionController._execute_eviction_action = l2_action
    BaseL1Manager._free_and_report_deleted = l1_removed
    _log = log
    return True
