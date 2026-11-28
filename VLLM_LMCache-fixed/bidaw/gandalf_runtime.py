# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded local-FS tiering for the pinned LMCache MP server."""
import asyncio
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import asdict, dataclass, replace
from enum import Enum, auto
from functools import wraps
import logging
import math
import os
from pathlib import Path
from threading import Lock
import time

from bidaw.gandalf_policy import Block, Decision, History, plan_evictions
from bidaw.l2_event_probe import l1_cycle_active, record_l1_cycle_phase, record_tier_event

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
    write_concurrency: int = 8

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
        concurrency = int(os.environ.get("GANDALF_L2_WRITE_CONCURRENCY", "8"))
        if (capacity <= 0 or horizon <= 0 or concurrency <= 0
                or not all(math.isfinite(x) for x in (alpha, recompute, drive_price, endurance, days))
                or min(recompute, drive_price, endurance) <= 0 or min(alpha, days) < 0):
            raise ValueError("Invalid Gandalf capacity, prediction horizon, write concurrency or cost settings")
        return cls(mode, capacity, alpha, recompute, drive_price / endurance,
                   endurance, days, horizon, concurrency)


async def _wait_for_io(future):
    """Delay task cancellation until the underlying I/O can no longer use its buffer."""
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            cancelled = True
        except Exception:
            break  # The caller handles the result and commits or rolls back accounting.
    return cancelled


class SourcePins:
    """Defer returning an exported DRAM buffer to the pool until I/O is done."""

    def __init__(self, memory_manager, success_code):
        self._lock = Lock()
        self._pins = Counter()
        self._deferred = {}
        self._free = memory_manager.free
        self._success_code = success_code
        memory_manager.free = self.free

    def free(self, objects):
        with self._lock:
            ready = []
            for obj in objects:
                identity = id(obj)
                if identity in self._pins:
                    self._deferred[identity] = obj
                else:
                    ready.append(obj)
            return self._free(ready) if ready else self._success_code

    @contextmanager
    def hold(self, obj):
        identity = id(obj)
        with self._lock:
            available = obj.is_valid() and identity not in self._deferred
            if available:
                self._pins[identity] += 1
        try:
            yield available
        finally:
            if available:
                with self._lock:
                    self._pins[identity] -= 1
                    if not self._pins[identity]:
                        del self._pins[identity]
                        deferred = self._deferred.pop(identity, None)
                        if deferred is not None:
                            self._free([deferred])


class BoundedFS:
    def __init__(self, adapter, settings, delete_files):
        self.adapter = adapter
        self.settings = settings
        self.delete_files = delete_files
        self.lock = Lock()
        self.io_lock = asyncio.Lock()
        self.residents = OrderedDict()
        self.readers = Counter()
        self._removing = set()
        self.counters = Counter()
        self.used = 0
        self.started_at = None
        self.inflight_bytes = 0
        self.inflight_keys = {}
        self._write_slots = asyncio.Semaphore(settings.write_concurrency)
        self._write_executor = ThreadPoolExecutor(
            max_workers=settings.write_concurrency, thread_name_prefix="gandalf-ssd-write",
        )
        self._active_io = set()
        self._closing = False
        self.source_pins = None

    async def write_object(self, key, obj, victims=None, *, can_move=None):
        # Native read leases can expire, and forced cache clears bypass them.
        # The pool pin additionally protects the raw memoryview in both cases.
        if self.source_pins is not None:
            with self.source_pins.hold(obj) as available:
                if not available:
                    return Admission.BLOCKED, 0
                with memoryview(obj.byte_array) as data:
                    return await self.admit(key, data, victims, can_move=can_move)
        with memoryview(obj.byte_array) as data:
            return await self.admit(key, data, victims, can_move=can_move)

    def begin(self):
        with self.lock:
            if self.started_at is None:
                self.started_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            return dict(self.residents), set(self.readers)

    def placement_snapshot(self):
        with self.lock:
            return dict(self.residents), set(self.readers), set(self.inflight_keys)

    async def drain(self):
        self._closing = True
        while self._active_io:
            await asyncio.wait(tuple(self._active_io))

    def drain_before_close(self):
        if self.adapter._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.drain(), self.adapter._loop).result()
        self._write_executor.shutdown(wait=True)

    def resident_sessions(self):
        with self.lock:
            return {key.cache_salt for key in self.residents.keys() | self.inflight_keys.keys()}

    def status(self):
        with self.lock:
            return {**self.counters, "used_bytes": self.used, "resident_objects": len(self.residents),
                    "inflight_bytes": self.inflight_bytes}

    def increment(self, field, value=1):
        with self.lock:
            self.counters[field] += value

    def touch(self, keys):
        with self.lock:
            for key in reversed(keys):
                if key in self.residents:
                    self.residents.move_to_end(key)

    async def pin_load(self, keys):
        while True:
            with self.lock:
                if not self._removing or self._removing.isdisjoint(keys):
                    self.readers.update(keys)
                    return
            # Only a read of a key being unlinked waits for the placement lock.
            async with self.io_lock:
                pass

    async def execute_load(self, keys, objects, task_id, fs_load):
        await self.pin_load(keys)
        try:
            return await fs_load(self.adapter, keys, objects, task_id)
        finally:
            self.unpin_load(keys)

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

    def hold_copy(self, key):
        with self.lock:
            if key not in self.residents or key in self._removing:
                return False
            self.readers[key] += 1
            return True

    async def _remove(self, keys):
        with self.lock:
            keys = [key for key in keys if key not in self.readers and key not in self.inflight_keys]
            self._removing.update(keys)
        try:
            removed, sizes = await self.delete_files(self.adapter, keys) if keys else ([], [])
            with self.lock:
                for key in removed:
                    self.used -= self.residents.pop(key, 0)
            return removed, sizes
        finally:
            with self.lock:
                self._removing.difference_update(keys)

    async def delete(self, keys):
        async with self.io_lock:
            return await self._remove(keys)

    def _write_file(self, key, data):
        final, temporary = self.adapter._key_to_file_and_tmp_path(key)
        started = time.monotonic_ns() if l1_cycle_active() else None
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
            if started is not None:
                record_l1_cycle_phase("ssd_write_ms", (time.monotonic_ns() - started) / 1e6)
            temporary.unlink(missing_ok=True)

    async def _reserve(self, key, size, victims, can_move):
        async with self.io_lock:
            if self._closing or (can_move is not None and not can_move()):
                return None, Admission.BLOCKED
            with self.lock:
                pending = self.inflight_keys.get(key)
                existing = self.residents.get(key)
                needed = self.used + self.inflight_bytes + size - self.settings.capacity
            if pending is not None:
                return pending, None
            if existing is not None:
                if existing == size and await self.adapter._key_exists_on_disk(key):
                    return None, Admission.STORED
                LOGGER.error("Tracked SSD object is missing or has inconsistent size: %s", key)
                return None, Admission.FAILED

            if victims is None:
                victims = []
                if needed > 0:
                    with self.lock:
                        for victim, resident_size in self.residents.items():
                            if victim not in self.readers:
                                victims.append(victim)
                                needed -= resident_size
                                if needed <= 0:
                                    break
                        if needed > 0 and self.inflight_keys:
                            # With write-through, a soon-to-be-resident object can
                            # become the next LRU victim. Retry after its commit.
                            return next(iter(self.inflight_keys.values())), None
            else:
                with self.lock:
                    if any(victim not in self.residents or victim in self.readers for victim in victims):
                        return None, Admission.BLOCKED
                    needed -= sum(self.residents[victim] for victim in victims)
            if needed > 0:
                return None, Admission.BLOCKED

            # Hold the placement lock only through unlink/metadata updates.
            # A failed child unlink must stop its parent's deletion.
            for victim in victims:
                removed, sizes = await self._remove([victim])
                if not removed:
                    return None, Admission.FAILED
                self.adapter._notify_keys_deleted(removed, sizes)
                self.increment("ssd_evictions")
                record_tier_event("ssd_eviction", victim, bytes=sizes[0])

            reservation = asyncio.get_running_loop().create_future()
            with self.lock:
                self.inflight_keys[key] = reservation
                self.inflight_bytes += size
            return reservation, Admission.STORED

    async def admit(self, key, data, victims=None, *, can_move=None):
        """Persist one object, retaining its source buffer through physical completion."""
        if self._closing or len(data) > self.settings.capacity:
            return Admission.BLOCKED, 0
        task = asyncio.current_task()
        self._active_io.add(task)
        try:
            return await self._admit(key, data, victims, can_move)
        finally:
            self._active_io.discard(task)

    async def _admit(self, key, data, victims, can_move):
        size = len(data)
        while True:
            # A concurrent writer of this key owns the temporary path. Only
            # deduplicate against the final file after it finishes.
            reserve = asyncio.create_task(self._reserve(key, size, victims, can_move))
            cancelled = await _wait_for_io(reserve)
            reservation, outcome = reserve.result()
            if reservation is not None and outcome is not None:
                break
            if cancelled:
                raise asyncio.CancelledError
            if reservation is None:
                return outcome, 0
            await asyncio.shield(reservation)

        written = None
        try:
            if cancelled:
                raise asyncio.CancelledError
            async with self._write_slots:
                loop = asyncio.get_running_loop()
                write = loop.run_in_executor(
                    self._write_executor, copy_context().run, self._write_file, key, data,
                )
                cancelled = await _wait_for_io(write)
                write.result()
                written = size
        except OSError:
            LOGGER.exception("SSD write failed for %s", key)
        finally:
            with self.lock:
                if written is not None:
                    self.residents[key] = size
                    self.used += size
                    self.counters["writes"] += 1
                    self.counters["write_bytes"] += size
                self.inflight_bytes -= size
                del self.inflight_keys[key]
            try:
                if written is not None:
                    self.adapter._notify_keys_stored([key], [size])
            except OSError:
                LOGGER.exception("SSD store notification failed for %s", key)
            finally:
                reservation.set_result(None)
        if cancelled:
            raise asyncio.CancelledError
        return (Admission.STORED, written) if written is not None else (Admission.FAILED, 0)


def _record_decision(decision, state, selection_ms):
    record_tier_event("tier_decision", decision.key, action=decision.action,
                      drop_cost=decision.drop_cost, write_cost=decision.write_cost,
                      replacement_cost=decision.replacement_cost, victims=len(decision.victims),
                      drop_score=decision.drop_score, move_score=decision.move_score,
                      linear_state=state, selection_ms=selection_ms)


def _release_l1(manager, budget, decision, success_code):
    key = decision.key
    released = manager.delete([key]).get(key) == success_code
    if released:
        if decision.action == "drop":
            budget.increment("drops")
        record_tier_event("tier_l1_release", key, action=decision.action)
    return released


async def _move(manager, budget, decision, success_code, can_move, prerequisites):
    for prior in prerequisites:
        if not await prior:
            budget.increment("blocked_migrations")
            record_tier_event("tier_migration_blocked", decision.key)
            return False
    key = decision.key
    reservation = manager.reserve_read([key]).get(key)
    if reservation is None or reservation[0] != success_code or reservation[1] is None:
        return False
    try:
        victims = decision.victims if budget.settings.mode == "gandalf" else None
        outcome, written = await budget.write_object(key, reservation[1], victims, can_move=can_move)
    except Exception:
        LOGGER.exception("L1 to SSD migration failed")
        outcome = Admission.FAILED
    finally:
        manager.finish_read([key])
    if outcome is Admission.FAILED:
        budget.increment("failed_migrations")
        record_tier_event("tier_migration_failed", key)
        return False
    if outcome is Admission.BLOCKED:
        budget.increment("blocked_migrations")
        record_tier_event("tier_migration_blocked", key)
        return False
    budget.increment("migrations" if written else "reused_ssd_copies")
    record_tier_event("tier_migration_complete", key, bytes=written)
    _release_l1(manager, budget, decision, success_code)
    return True


def _move_sessions(decision):
    return {decision.key.cache_salt} | {key.cache_salt for key in decision.victims}


async def _move_batch(manager, budget, moves, success_code):
    tasks = []
    latest = {}
    for decision, can_move in moves:
        # Gandalf's plan orders ancestors before descendants and SSD victim
        # leaves before parents. Only conflicting sessions share a dependency.
        scopes = _move_sessions(decision) if budget.settings.mode == "gandalf" else ()
        prerequisites = {latest[scope] for scope in scopes if scope in latest}
        task = asyncio.create_task(_move(manager, budget, decision, success_code,
                                         can_move, prerequisites))
        tasks.append(task)
        latest.update(dict.fromkeys(scopes, task))
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result


def reclaim_batch(manager, budget, plan, success_code):
    moves = []
    unsafe = set()
    held = []
    try:
        for decision, state, selection_ms, can_move in plan:
            _record_decision(decision, state, selection_ms)
            key = decision.key
            if decision.action == "move":
                moves.append((decision, can_move))
            elif key.cache_salt not in unsafe:
                if decision.action == "release_duplicate":
                    if not budget.hold_copy(key):
                        unsafe.add(key.cache_salt)
                        continue
                    held.append(key)
                    budget.increment("reused_ssd_copies")
                    record_tier_event("tier_migration_complete", key, bytes=0)
                if not _release_l1(manager, budget, decision, success_code):
                    unsafe.add(key.cache_salt)
        moves = [(decision, can_move) for decision, can_move in moves
                 if _move_sessions(decision).isdisjoint(unsafe)]
        if moves:
            operation = _move_batch(manager, budget, moves, success_code)
            try:
                future = asyncio.run_coroutine_threadsafe(operation, budget.adapter._loop)
            except RuntimeError:
                operation.close()
                raise
            future.result()
    finally:
        budget.unpin_load(held)


def reclaim(manager, budget, decision, success_code, *, linear_state=None,
            selection_ms=None, can_move=None):
    reclaim_batch(manager, budget, [(decision, linear_state, selection_ms, can_move)], success_code)


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


def _l1_eviction_snapshot(manager, *, include_staging):
    with manager._lock:
        dram = {}
        eligible = set()
        for key, entry in manager._objects.items():
            dram[key] = entry.memory_obj.get_size()
            # BaseL1Manager.is_key_evictable accepts an unlocked resident OR
            # any expired staging write lease. delete() checks both locks again.
            if not entry.read_lock.is_locked() or any(
                not stage.write_lock.is_locked()
                for stage in manager._staging.get(key, {}).values()
            ):
                eligible.add(key)
        staging = tuple(manager._staging) if include_staging else ()
    return dram, eligible, staging


def apply_tiering():
    settings = Settings.from_env()
    if settings is None:
        return
    if os.environ.get("BIDAW_PAD_TAIL") == "1":
        raise ValueError("The first-stage benchmark requires BIDAW_PAD_TAIL=0")

    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.internal_api import L1ManagerListener
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

    class LinearCommitListener(L1ManagerListener):
        def on_l1_keys_write_finished(self, keys):
            history.commit_linear(keys)

        def _ignore(self, keys):
            pass

        on_l1_keys_reserved_read = _ignore
        on_l1_keys_read_finished = _ignore
        on_l1_keys_reserved_write = _ignore
        on_l1_keys_finish_write_and_reserve_read = _ignore
        on_l1_keys_deleted_by_manager = _ignore
        on_l1_keys_accessed = _ignore

    fs_init = FSL2Adapter.__init__
    fs_delete = FSL2Adapter._execute_delete
    fs_load = FSL2Adapter._execute_load
    fs_touch = FSL2Adapter._notify_keys_accessed
    fs_close = FSL2Adapter.close
    manager_close = StorageManager.close
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
        manager = self._eviction_controllers[0]._l1_manager
        self._gandalf_budget.source_pins = SourcePins(manager._memory_manager, L1Error.SUCCESS)
        self._eviction_controllers[0]._gandalf_budget = self._gandalf_budget
        if settings.mode == "gandalf":
            self._eviction_controllers[0]._l1_manager.register_listener(LinearCommitListener())
        _STORAGE = self

    def close_fs(self):
        self._gandalf_budget.drain_before_close()
        return fs_close(self)

    def close_manager(self):
        # StoreController.stop releases its in-flight source pins. Drain and
        # reject new admissions before native shutdown is allowed to do that.
        for adapter in self._l2_adapters.values():
            if hasattr(adapter, "_gandalf_budget"):
                adapter._gandalf_budget.drain_before_close()
        return manager_close(self)

    async def execute_store(self, keys, objects, task_id):
        transferred = 0
        success = True
        budget = self._gandalf_budget

        writes = (budget.write_object(key, obj) for key, obj in zip(keys, objects, strict=True))
        results = await asyncio.gather(*writes, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                LOGGER.error("Bounded FS store task failed", exc_info=result)
                budget.increment("failed_writes")
                success = False
                continue
            outcome, written = result
            transferred += written
            if outcome is Admission.FAILED:
                budget.increment("failed_writes")
                success = False
            elif outcome is Admission.BLOCKED:
                budget.increment("blocked_writes")
                success = False
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, transferred)
        self._store_efd.notify()

    async def execute_delete(self, keys):
        return await self._gandalf_budget.delete(keys)

    async def execute_load(self, keys, objects, task_id):
        return await self._gandalf_budget.execute_load(keys, objects, task_id, fs_load)

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
                plan = [(Decision(key, "move"), None, None, None) for key in action.keys]
            else:
                dram, eligible, staging = _l1_eviction_snapshot(
                    manager, include_staging=history.gc_due(),
                )
                disk, pinned, in_flight = budget.placement_snapshot()
                in_flight.update(staging)
                decisions = plan_evictions(
                    dram, eligible, disk, pinned, settings.capacity, history,
                    settings.recompute_price, budget.write_price, settings.alpha,
                    len(action.keys), in_flight=in_flight,
                )
                plan = []
                for _ in action.keys:
                    started = time.perf_counter()
                    decision = next(decisions, None)
                    selection_ms = (time.perf_counter() - started) * 1000
                    record_l1_cycle_phase("selection_ms", selection_ms)
                    if decision is None:
                        break
                    state = decision.linear_state
                    can_move = None
                    if state == "live" and decision.action == "move":
                        can_move = lambda key=decision.key: history.linear_state(key) == "live"
                    plan.append((decision, state, selection_ms, can_move))
            reclaim_batch(manager, budget, plan, L1Error.SUCCESS)
        except Exception:
            LOGGER.exception("Tier eviction cycle failed")
            budget.increment("eviction_failures")
        finally:
            budget.increment("active_evictions", -1)

    FSL2Adapter.__init__ = init_fs
    FSL2Adapter.close = close_fs
    FSL2Adapter._execute_store = execute_store
    FSL2Adapter._execute_delete = execute_delete
    FSL2Adapter._execute_load = execute_load
    FSL2Adapter._notify_keys_accessed = touch
    StorageManager.__init__ = init_manager
    StorageManager.close = close_manager
    L1EvictionController.execute_eviction_action = evict
    if settings.mode == "gandalf":
        _observe_prefixes(history)


def _group_kinds(attention):
    kinds = attention.group_kinds or ("attention",) * len(attention.num_chunks_in_sw)
    return {group: "full" if kind == "attention" and window < 0 else "linear"
            for group, (kind, window) in enumerate(zip(kinds, attention.num_chunks_in_sw, strict=True))}


def _observe_prefixes(history):
    from lmcache.v1.distributed.api import ipc_key_to_object_keys
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    resolve = MPCacheServerContext.resolve_obj_keys
    lookup = LookupModule.lookup

    @wraps(resolve)
    def resolve_keys(self, key, object_group_ids):
        if key.world_size != 1 or not key.cache_salt:
            raise ValueError("Gandalf requires one rank and salted sessions")
        result = resolve(self, key, object_group_ids)
        attention = self.layout_desc_registry.find_attn_desc(key.model_name, key.world_size)
        round_number = history.observe(key.cache_salt, key.request_id)
        start = key.start // self.chunk_size
        kinds = _group_kinds(attention)
        history.register_layout(key.model_name, kinds)
        needs_prefix = any(kinds[group] == "full" or attention.num_chunks_in_sw[group] > 1
                           for group in object_group_ids)
        prefix_hashes = (self.token_hasher.compute_packed_chunk_hashes(key.token_bytes, end=key.start)
                         if start and needs_prefix else [])
        for group, keys in zip(object_group_ids, result, strict=True):
            window = attention.num_chunks_in_sw[group]
            if kinds[group] == "full":
                prefix = ipc_key_to_object_keys(key, prefix_hashes, [group])[0] if start else []
                history.bind(prefix + keys, round_number=round_number, group=group)
            else:
                if window > 1:
                    hashes = prefix_hashes + [obj_key.identity.chunk_hash for obj_key in keys]
                for boundary, obj_key in enumerate(keys, start=start + 1):
                    window_hashes = tuple(hashes[max(0, boundary - window):boundary]) if window > 1 else ()
                    # Native STORE completion round-trips untyped attributes
                    # through MessagePack before invoking the L1 listener.
                    obj_key.extra_attributes["gandalf_block"] = asdict(Block(
                        "linear", key.cache_salt, round_number, group, boundary,
                        max(1, window), window_hashes,
                    ))
        return result

    @wraps(lookup)
    def observe(self, key, tp_size):
        if key.world_size != 1 or not key.cache_salt:
            raise ValueError("Gandalf requires one rank and salted replay sessions")
        history.observe(key.cache_salt, key.request_id)
        registry = self._ctx.layout_desc_registry
        if registry.find(key.model_name, key.world_size) is not None:
            attention = registry.find_attn_desc(key.model_name, key.world_size)
            history.register_layout(key.model_name, _group_kinds(attention))
        storage = self._ctx.storage_manager
        budget = storage._gandalf_budget
        budget.begin()
        if history.gc_due():
            # Also collect on quiet workloads that rarely cross the watermark.
            manager = storage._eviction_controllers[0]._l1_manager
            with manager._lock:
                sessions = {obj.cache_salt for obj in manager._objects}
                sessions.update(obj.cache_salt for obj in manager._staging)
            history.prune(sessions | budget.resident_sessions())
        return lookup(self, key, tp_size)

    MPCacheServerContext.resolve_obj_keys = resolve_keys
    LookupModule.lookup = observe
