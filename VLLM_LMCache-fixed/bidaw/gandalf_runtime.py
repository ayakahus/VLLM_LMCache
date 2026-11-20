# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded local-FS tiering for the pinned LMCache MP server."""
import asyncio
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import wraps
import logging
import math
import os
from pathlib import Path
from threading import Lock
import time

from bidaw.gandalf_policy import Decision, History, choose
from bidaw.l2_event_probe import record_tier_event

LOGGER = logging.getLogger(__name__)
_STORAGE = None


class Admission(Enum):
    STORED = auto()
    BLOCKED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class Settings:
    mode: str
    capacity: int
    alpha: float
    recompute_price: float
    write_price: float
    endurance_bytes: float
    target_days: float
    horizon: int

    @classmethod
    def from_env(cls):
        mode = os.environ.get("GANDALF_TIER_MODE")
        if mode is None:
            return None
        if mode not in {"write_through", "lru_writeback", "gandalf"}:
            raise ValueError(f"Unknown GANDALF_TIER_MODE: {mode}")
        capacity = int(os.environ.get("GANDALF_L2_BYTES", str(2 << 30)))
        alpha = float(os.environ.get("GANDALF_ALPHA", "1"))
        recompute = float(os.environ.get("GANDALF_RECOMPUTE_COST", "0.00002"))
        drive_price = float(os.environ.get("GANDALF_SSD_PRICE", "100"))
        endurance = float(os.environ.get("GANDALF_SSD_TBW_BYTES", "6e14"))
        days = float(os.environ.get("GANDALF_TARGET_DAYS", "0"))
        horizon = int(os.environ.get("GANDALF_RETURN_HORIZON", "128"))
        if (capacity <= 0 or horizon <= 0
                or not all(math.isfinite(x) for x in (alpha, recompute, drive_price, endurance, days))
                or min(recompute, drive_price, endurance) <= 0 or min(alpha, days) < 0):
            raise ValueError("Invalid Gandalf capacity, prediction horizon or cost settings")
        return cls(mode, capacity, alpha, recompute, drive_price / endurance,
                   endurance, days, horizon)


class BoundedFS:
    def __init__(self, adapter, settings, delete_files):
        self.adapter = adapter
        self.settings = settings
        self.delete_files = delete_files
        self.lock = Lock()
        self.io_lock = asyncio.Lock()
        self.residents = OrderedDict()
        self.readers = Counter()
        self.counters = Counter()
        self.used = 0
        self.started_at = None

    def begin(self):
        with self.lock:
            if self.started_at is None:
                self.started_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            return dict(self.residents), set(self.readers)

    def status(self):
        with self.lock:
            return {**self.counters, "used_bytes": self.used, "resident_objects": len(self.residents)}

    def increment(self, field, value=1):
        with self.lock:
            self.counters[field] += value

    def touch(self, keys):
        with self.lock:
            for key in reversed(keys):
                if key in self.residents:
                    self.residents.move_to_end(key)

    def pin_load(self, keys):
        with self.lock:
            self.readers.update(keys)

    def unpin_load(self, keys):
        with self.lock:
            for key in keys:
                self.readers[key] -= 1
                if not self.readers[key]:
                    del self.readers[key]

    def write_price(self, size):
        with self.lock:
            written = self.counters["write_bytes"]
            started = self.started_at
        if not self.settings.target_days or started is None:
            return self.settings.write_price
        elapsed = max(1.0, time.monotonic() - started)
        budget = self.settings.endurance_bytes * elapsed / (86400 * self.settings.target_days)
        multiplier = (written + size) / max(size, budget)
        return self.settings.write_price * multiplier

    async def _remove(self, keys):
        removed, sizes = await self.delete_files(self.adapter, keys)
        with self.lock:
            for key in removed:
                self.used -= self.residents.pop(key, 0)
        return removed, sizes

    async def delete(self, keys):
        async with self.io_lock:
            return await self._remove(keys)

    def _write_file(self, key, data):
        final, temporary = self.adapter._key_to_file_and_tmp_path(key)
        try:
            if self.adapter._use_odirect:
                self.adapter._write_with_odirect(temporary, data)
            else:
                with temporary.open("wb") as stream:
                    if stream.write(data) != len(data):
                        raise OSError(f"Short write to {temporary}")
            if temporary.stat().st_size != len(data):
                raise OSError(f"Unexpected size for {temporary}")
            temporary.replace(final)
        finally:
            temporary.unlink(missing_ok=True)

    async def admit(self, key, data, victims=None):
        """Persist one object and return the outcome and newly written bytes."""
        async with self.io_lock:
            disk, pinned = self.snapshot()
            if key in disk:
                if disk[key] == len(data) and await self.adapter._key_exists_on_disk(key):
                    return Admission.STORED, 0
                LOGGER.error("Tracked SSD object is missing or has inconsistent size: %s", key)
                return Admission.FAILED, 0
            needed = sum(disk.values()) + len(data) - self.settings.capacity
            if len(data) > self.settings.capacity:
                return Admission.BLOCKED, 0
            if victims is None:
                # The two LRU-based L2 baselines evict oldest available objects.
                plan = []
                for victim, size in disk.items():
                    if needed <= 0:
                        break
                    if victim not in pinned:
                        plan.append(victim)
                        needed -= size
                victims = plan
            else:
                if any(victim not in disk or victim in pinned for victim in victims):
                    return Admission.BLOCKED, 0
                needed -= sum(disk[victim] for victim in victims)
            if needed > 0:
                return Admission.BLOCKED, 0

            # Delete leaves in plan order. A failed child removal stops the
            # plan before its ancestor can be removed.
            for victim in victims:
                with self.lock:
                    if victim in self.readers:
                        return Admission.BLOCKED, 0
                removed, sizes = await self._remove([victim])
                if not removed:
                    return Admission.FAILED, 0
                self.adapter._notify_keys_deleted(removed, sizes)
                self.increment("ssd_evictions")
                record_tier_event("ssd_eviction", victim, bytes=sizes[0])

            try:
                # The immutable snapshot remains valid even if an async task is cancelled.
                await asyncio.to_thread(self._write_file, key, data)
            except OSError:
                LOGGER.exception("SSD write failed for %s", key)
                return Admission.FAILED, 0
            with self.lock:
                self.residents[key] = len(data)
                self.used += len(data)
                self.counters["writes"] += 1
                self.counters["write_bytes"] += len(data)
            self.adapter._notify_keys_stored([key], [len(data)])
            return Admission.STORED, len(data)


def reclaim(manager, budget, decision, success_code):
    key = decision.key
    record_tier_event("tier_decision", key, action=decision.action,
                      drop_cost=decision.drop_cost, write_cost=decision.write_cost,
                      replacement_cost=decision.replacement_cost, victims=len(decision.victims),
                      drop_score=decision.drop_score, move_score=decision.move_score)
    if decision.action != "drop":
        reservation = manager.reserve_read([key]).get(key)
        if reservation is None or reservation[0] != success_code or reservation[1] is None:
            return
        try:
            data = bytes(reservation[1].byte_array)
            victims = decision.victims if budget.settings.mode == "gandalf" else None
            operation = budget.admit(key, data, victims)
            try:
                future = asyncio.run_coroutine_threadsafe(operation, budget.adapter._loop)
            except RuntimeError:
                operation.close()
                raise
            outcome, written = future.result()
        except Exception:
            LOGGER.exception("L1 to SSD migration failed")
            outcome = Admission.FAILED
        finally:
            manager.finish_read([key])
        if outcome is Admission.FAILED:
            budget.increment("failed_migrations")
            record_tier_event("tier_migration_failed", key)
            return
        if outcome is Admission.BLOCKED:
            budget.increment("blocked_migrations")
            record_tier_event("tier_migration_blocked", key)
            return
        budget.increment("migrations" if written else "reused_ssd_copies")
        record_tier_event("tier_migration_complete", key, bytes=written)
    released = manager.delete([key]).get(key) == success_code
    if released:
        if decision.action == "drop":
            budget.increment("drops")
        record_tier_event("tier_l1_release", key, action=decision.action)


def status():
    if _STORAGE is None:
        return None
    budget = _STORAGE._gandalf_budget
    stores = next(iter(_STORAGE._store_controllers.values())).report_status()
    _, l1_capacity = _STORAGE._eviction_controllers[0]._l1_manager.get_memory_usage()
    return {**budget.status(), "mode": budget.settings.mode,
            "capacity_bytes": budget.settings.capacity, "alpha": budget.settings.alpha,
            "l1_capacity_bytes": l1_capacity,
            "pending_keys": stores["pending_keys_count"],
            "in_flight_stores": stores["in_flight_task_count"]}


def apply_tiering():
    settings = Settings.from_env()
    if settings is None:
        return
    if os.environ.get("BIDAW_PAD_TAIL") == "1":
        raise ValueError("The first-stage benchmark requires BIDAW_PAD_TAIL=0")

    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        FSL2Adapter, FSL2AdapterConfig, L2StoreResult,
    )
    from lmcache.v1.distributed.storage_controllers.eviction_controller import L1EvictionController
    from lmcache.v1.distributed.storage_controllers.store_policy import (
        DefaultStorePolicy, register_store_policy,
    )
    from lmcache.v1.distributed.storage_manager import StorageManager

    class DeferredStorePolicy(DefaultStorePolicy):
        def select_store_targets(self, keys, adapters):
            return {}

    register_store_policy("gandalf_deferred", DeferredStorePolicy)
    history = History(settings.horizon)
    fs_init = FSL2Adapter.__init__
    fs_delete = FSL2Adapter._execute_delete
    fs_load = FSL2Adapter._execute_load
    fs_touch = FSL2Adapter._notify_keys_accessed
    manager_init = StorageManager.__init__
    native_evict = L1EvictionController.execute_eviction_action

    def init_fs(self, config):
        directory = Path(config.base_path)
        if config.shared or (directory.exists() and any(directory.iterdir())):
            raise ValueError("Bounded FS tiering requires a dedicated empty local L2 directory")
        fs_init(self, config)
        self._gandalf_budget = BoundedFS(self, settings, fs_delete)

    def init_manager(self, config):
        global _STORAGE
        adapters = config.l2_adapter_config.adapters
        if len(adapters) != 1 or not isinstance(adapters[0], FSL2AdapterConfig):
            raise ValueError("Tier comparison requires one local FS adapter")
        if settings.mode != "write_through":
            config = replace(config, store_policy="gandalf_deferred")
        manager_init(self, config)
        adapter = next(iter(self._l2_adapters.values()))
        if (len(self._eviction_controllers) != 1 or not isinstance(adapter, FSL2Adapter)
                or not isinstance(self._eviction_controllers[0]._l1_manager, DramL1Manager)):
            self.close()
            raise ValueError("Tier comparison requires one DRAM manager and one FS adapter")
        self._gandalf_budget = adapter._gandalf_budget
        self._eviction_controllers[0]._gandalf_budget = self._gandalf_budget
        _STORAGE = self

    async def execute_store(self, keys, objects, task_id):
        transferred = 0
        success = True
        budget = self._gandalf_budget
        try:
            for key, obj in zip(keys, objects, strict=True):
                outcome, written = await budget.admit(key, bytes(obj.byte_array))
                transferred += written
                if outcome is Admission.FAILED:
                    budget.increment("failed_writes")
                    success = False
                elif outcome is Admission.BLOCKED:
                    budget.increment("blocked_writes")
                    success = False
        except Exception:
            LOGGER.exception("Bounded FS store task failed")
            budget.increment("failed_writes")
            success = False
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, transferred)
        self._store_efd.notify()

    async def execute_delete(self, keys):
        return await self._gandalf_budget.delete(keys)

    async def execute_load(self, keys, objects, task_id):
        budget = self._gandalf_budget
        budget.pin_load(keys)
        try:
            return await fs_load(self, keys, objects, task_id)
        finally:
            budget.unpin_load(keys)

    def touch(self, keys):
        self._gandalf_budget.touch(keys)
        return fs_touch(self, keys)

    def evict(self, action):
        if settings.mode == "write_through":
            return native_evict(self, action)
        budget = getattr(self, "_gandalf_budget", None)
        if budget is None:
            return  # The native eviction thread starts before the adapter is attached.
        manager = self._l1_manager
        budget.increment("active_evictions")
        try:
            if settings.mode == "lru_writeback":
                for key in action.keys:
                    reclaim(manager, budget, Decision(key, "move"), L1Error.SUCCESS)
                return

            attempted = set()
            for _ in action.keys:
                with manager._lock:
                    dram = {key: entry.memory_obj.get_size() for key, entry in manager._objects.items()}
                eligible = {key for key in dram.keys() - attempted if manager.is_key_evictable(key)}
                disk, pinned = budget.snapshot()
                if not eligible:
                    break
                decision = choose(dram, eligible, disk, pinned, settings.capacity, history,
                                  settings.recompute_price, budget.write_price, settings.alpha)
                if decision is None:
                    break
                attempted.add(decision.key)
                reclaim(manager, budget, decision, L1Error.SUCCESS)
        except Exception:
            LOGGER.exception("Tier eviction cycle failed")
            budget.increment("eviction_failures")
        finally:
            budget.increment("active_evictions", -1)

    FSL2Adapter.__init__ = init_fs
    FSL2Adapter._execute_store = execute_store
    FSL2Adapter._execute_delete = execute_delete
    FSL2Adapter._execute_load = execute_load
    FSL2Adapter._notify_keys_accessed = touch
    StorageManager.__init__ = init_manager
    L1EvictionController.execute_eviction_action = evict
    if settings.mode == "gandalf":
        _observe_prefixes(history)


def _observe_prefixes(history):
    from lmcache.v1.distributed.api import ipc_key_to_object_keys
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    resolve = MPCacheServerContext.resolve_obj_keys
    lookup = LookupModule.lookup

    @wraps(resolve)
    def resolve_keys(self, key, object_group_ids):
        if key.world_size != 1 or list(object_group_ids) != [0] or not key.cache_salt:
            raise ValueError("Gandalf requires one full-attention KV group, one rank and salted sessions")
        if key.start:
            # STORE may only include a suffix in its range; reconstruct its ancestors.
            hashes = self.token_hasher.compute_packed_chunk_hashes(key.token_bytes, end=key.end)
            history.bind(ipc_key_to_object_keys(key, hashes, [0])[0])
            return resolve(self, key, object_group_ids)
        result = resolve(self, key, object_group_ids)
        history.bind(result[0])
        return result

    @wraps(lookup)
    def observe(self, key, tp_size):
        if key.world_size != 1 or not key.cache_salt:
            raise ValueError("Gandalf requires one rank and salted replay sessions")
        registry = self._ctx.layout_desc_registry
        if registry.find(key.model_name, key.world_size) is not None:
            attention = registry.find_attn_desc(key.model_name, key.world_size)
            if (tuple(attention.num_chunks_in_sw) != (-1,)
                    or attention.group_kinds not in ((), ("attention",))):
                raise ValueError("Gandalf requires one full-attention KV group")
        history.observe(key.cache_salt, key.request_id)
        self._ctx.storage_manager._gandalf_budget.begin()
        return lookup(self, key, tp_size)

    MPCacheServerContext.resolve_obj_keys = resolve_keys
    LookupModule.lookup = observe
