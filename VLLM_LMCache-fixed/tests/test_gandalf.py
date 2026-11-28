import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field
import errno
import mmap
import os
from pathlib import Path
import tempfile
from threading import Event, Lock, Thread, current_thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bidaw.gandalf_policy import Block, Decision, History, choose, plan_evictions
from bidaw.gandalf_runtime import (
    Admission, BoundedFS, Settings, SourcePins, _group_kinds, _l1_eviction_snapshot,
    _read_one_object, reclaim,
)


@dataclass(frozen=True)
class Key:
    name: str
    cache_salt: str = "session"


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.history = History(8)
        self.root, self.child = Key("root"), Key("child")
        self.history.bind([self.root, self.child])

    def decision(self, dram, disk, capacity=4, fee=.01, pinned=(), eligible=None):
        return choose(dram, set(dram) if eligible is None else eligible, disk,
                      set(pinned), capacity, self.history, 1, lambda _: fee, 1)

    def test_moves_root_first_and_discards_leaf_first(self):
        dram = {self.child: 2, self.root: 2}
        move = self.decision(dram, {})
        self.assertEqual((move.action, move.key), ("move", self.root))
        drop = self.decision(dram, {}, fee=10)
        self.assertEqual((drop.action, drop.key), ("drop", self.child))
        self.assertEqual(self.decision({self.child: 2}, {self.root: 2}).key, self.child)

    def test_replacement_plan_prices_all_victims_and_preserves_ancestors(self):
        victim, victim_child = Key("victim", "old"), Key("victim_child", "old")
        new, new_child, extra = Key("new", "incoming"), Key("new_child", "incoming"), Key("extra", "incoming")
        self.history.bind([victim, victim_child])
        self.history.bind([self.root, new, new_child, extra])
        dram = {new: 3, new_child: 1, extra: 1, self.root: 2}
        disk = {self.root: 1, victim: 2, victim_child: 1}
        decision = self.decision(dram, disk, eligible={new})
        self.assertEqual(decision.victims, (victim_child, victim))
        self.assertAlmostEqual(decision.replacement_cost, 1)
        self.assertAlmostEqual(decision.move_score, .01 + 1 / 3)
        self.assertIsNone(self.decision(dram, disk, eligible={new}, pinned={victim_child}))

    def test_free_victim_does_not_hide_cost_of_second_victim(self):
        duplicate = Key("duplicate")
        victim = Key("victim")
        disk = {duplicate: 1, victim: 1}
        decision = self.decision({self.root: 2, duplicate: 1}, disk,
                                 capacity=2, eligible={self.root})
        self.assertEqual((decision.action, decision.key), ("drop", self.root))
        self.assertAlmostEqual(decision.move_score, .01 + .5 / 2)

    def test_other_full_candidate_can_win_when_older_candidate_is_blocked_or_costly(self):
        old = Key("old", "old")
        new = Key("new", "new")
        duplicate = Key("duplicate")
        victim = Key("victim")
        for key in (old, new, duplicate, victim):
            self.history.bind([key])
        dram = {old: 2, new: 1, duplicate: 1}
        disk = {victim: 2, duplicate: 1}
        with patch("bidaw.gandalf_policy._estimate_activity", return_value={"old": (10, .5), "new": (1, .5)}):
            for pinned in ((victim,), ()):
                with self.subTest(pinned=pinned):
                    decision = self.decision(dram, disk, capacity=3, pinned=pinned,
                                             eligible={old, new})
                    self.assertEqual((decision.action, decision.key), ("move", new))
                    self.assertEqual(decision.victims, (duplicate,))

    def test_move_gate_prefers_lower_cost_cold_root(self):
        cold = Key("cold-root", "cold")
        cold_child = Key("cold-child", "cold")
        self.history.bind([cold, cold_child])
        dram = dict.fromkeys((self.root, self.child, cold, cold_child), 1)
        with patch("bidaw.gandalf_policy._estimate_activity", return_value={"session": (1, .8), "cold": (20, .8)}):
            decision = self.decision(dram, {}, fee=.2)
        self.assertEqual((decision.action, decision.key), ("move", cold))
        self.assertAlmostEqual(decision.move_score, .01)
        self.assertAlmostEqual(decision.drop_score, .04)

    def test_cross_object_gate_drops_cheaper_leaf_and_resolves_ties_to_drop(self):
        cold = Key("cold-root", "cold")
        cold_leaf = Key("cold-leaf", "cold")
        self.history.bind([cold, cold_leaf])
        dram = dict.fromkeys((self.root, self.child, cold, cold_leaf), 1)
        eligible = {self.root, cold_leaf}
        with patch("bidaw.gandalf_policy._estimate_activity", return_value={"session": (1, .8), "cold": (20, .2)}):
            for fee in (.2, .01):
                with self.subTest(fee=fee):
                    decision = self.decision(dram, {}, fee=fee, eligible=eligible)
                    self.assertEqual((decision.action, decision.key), ("drop", cold_leaf))
                    self.assertAlmostEqual(decision.drop_score, .01)
                    self.assertAlmostEqual(decision.move_score, fee)

    def test_hot_l1_duplicate_competes_with_the_cold_leaf(self):
        duplicate = Key("duplicate", "hot")
        self.history.bind([duplicate])
        dram = dict.fromkeys((self.root, self.child, duplicate), 1)
        with patch("bidaw.gandalf_policy._estimate_activity", return_value={"session": (20, .2), "hot": (1, .8)}):
            decision = self.decision(dram, {duplicate: 1}, fee=1)
            retained_ssd = self.decision(dram, {duplicate: 1}, eligible={duplicate})
        self.assertEqual((decision.action, decision.key), ("drop", self.child))
        self.assertEqual((retained_ssd.action, retained_ssd.key), ("release_duplicate", duplicate))
        self.assertEqual(retained_ssd.write_cost, 0)

    def test_ssd_victim_plan_cannot_evict_internal_duplicate_around_pinned_leaf(self):
        incoming = Key("incoming", "new")
        incoming_child = Key("incoming-child", "new")
        self.history.bind([incoming, incoming_child])
        dram = {incoming: 2, incoming_child: 1, self.root: 1}
        disk = {self.root: 1, self.child: 1}
        decision = self.decision(dram, disk, capacity=2, eligible={incoming}, pinned={self.child})
        self.assertIsNone(decision)

    def test_request_clock_deduplicates_retries(self):
        self.history.observe("session", "1")
        self.history.observe("other", "2")
        self.history.observe("session", "1")
        self.history.observe("session", "3")
        self.assertEqual(self.history.clock, 3)
        self.assertEqual(list(self.history.gaps), [2])


    def test_activity_snapshot_matches_scan_without_holding_history_lock(self):
        for i, session in enumerate(("session", "b", "c", "session", "b", "b", "d", "session")):
            self.history.observe(session, i)
        clock = self.history.clock
        ages = {session: max(1, clock - last) for session, last in self.history.last.items()}
        gaps = tuple(self.history.gaps)
        observed = Counter(ages.values())
        expected = {}
        for session, age in ages.items():
            limit = age + self.history.horizon
            returns = sum(age < gap <= limit for gap in gaps)
            samples = sum(gap > age for gap in gaps)
            samples += sum(n for idle, n in observed.items() if idle >= limit)
            expected[session] = (age, (returns + 1) / (samples + 2))

        estimating = Event()
        changed = Event()
        from bidaw.gandalf_policy import _estimate_activity

        def estimate(*args):
            estimating.set()
            self.assertTrue(changed.wait(2))
            return _estimate_activity(*args)

        def update():
            if estimating.wait(2):
                self.history.observe("new", "new-round")
                self.history.bind([Key("new", "new")])
                changed.set()

        updater = Thread(target=update)
        updater.start()
        try:
            with patch("bidaw.gandalf_policy._estimate_activity", side_effect=estimate):
                snapshot = self.history.eviction_snapshot({self.root, self.child, Key("new", "new")})
        finally:
            updater.join(3)
        self.assertEqual(snapshot.activity, expected)
        self.assertEqual(snapshot.parents, {self.root: None, self.child: self.root})

    def test_batches_ancestors_before_descendants_without_reusing_ssd_victims(self):
        victim, leaf = Key("victim", "victim"), Key("leaf", "victim")
        self.history.bind([victim, leaf])
        dram = {self.root: 1, self.child: 1}
        with patch.object(self.history, "eviction_snapshot",
                          wraps=self.history.eviction_snapshot) as snapshot:
            results = list(plan_evictions(dram, set(dram), {victim: 1, leaf: 1},
                                          set(), 2, self.history, 1, lambda _: .01,
                                          0, 2))
        self.assertEqual([(d.key, d.action, d.victims) for d in results],
                         [(self.root, "move", (leaf,)), (self.child, "move", (victim,))])
        snapshot.assert_called_once()

    def test_idle_history_gc_preserves_ancestors_of_resident_objects(self):
        history = History(2, idle_window=2)
        root, child = self.root, self.child
        gone = Key("gone", "gone")
        history.observe("session", 1)
        history.bind([root, child])
        history.observe("gone", 2)
        history.bind([gone])
        for i in range(3, 7):
            history.observe("busy", i)
        self.assertTrue(history.gc_due())
        history.prune({child.cache_salt})
        self.assertFalse(history.gc_due())
        self.assertNotIn(gone, history.blocks)
        self.assertNotIn("gone", history.rounds)
        self.assertEqual(history.parents[child], root)
        self.assertIn(root, history.blocks)


@dataclass(frozen=True)
class HybridKey:
    name: str
    cache_salt: str = "session"
    extra_attributes: dict = field(default_factory=dict, compare=False, hash=False)
    group: int = 0

    @property
    def identity(self):
        return SimpleNamespace(model_name="hybrid", kv_rank=0, chunk_hash=self.name)


class HybridPlacementTests(unittest.TestCase):
    def setUp(self):
        self.history = History(8)
        self.history.register_layout("hybrid", {0: "full", 1: "linear"})

    def linear(self, boundary, round_number, *, session="session", window=1, hashes=()):
        key = HybridKey(str(boundary), session, {
            "gandalf_block": asdict(Block("linear", session, round_number, 1, boundary, window, list(hashes))),
        }, group=1)
        return key

    def test_classifies_recurrent_and_sliding_window_groups(self):
        layout = SimpleNamespace(group_kinds=("attention", "recurrent", "attention"),
                                 num_chunks_in_sw=[-1, 1, 3])
        self.assertEqual(_group_kinds(layout), {0: "full", 1: "linear", 2: "linear"})
        layout.group_kinds = ()
        self.assertEqual(_group_kinds(layout), {0: "full", 1: "linear", 2: "linear"})

    def test_expired_linear_drops_for_free_even_with_ssd_copy(self):
        old, new = self.linear(2, 1), self.linear(4, 2)
        self.history.commit_linear([old, new, old])
        self.assertEqual(self.history.linear_state(new), "live")
        self.assertEqual(self.history.linear_state(old), "superseded")
        self.assertNotIn(old, self.history.parents)
        decision = choose({new: 1, old: 1}, {old, new}, {old: 1}, set(), 4,
                          self.history, 1, lambda _: 0, 1)
        self.assertEqual(decision, Decision(old, "drop", drop_score=0.0))

    def test_live_sliding_window_retains_all_chunks_at_latest_boundary(self):
        old, left, right = (self.linear(2, 1, window=2), self.linear(3, 2, window=2),
                            self.linear(4, 2, window=2))
        self.history.commit_linear([old, right, left])
        self.assertEqual([self.history.linear_state(k) for k in (old, left, right)],
                         ["superseded", "live", "live"])

    def test_sliding_window_preserves_overlap_referenced_by_new_round(self):
        left = self.linear(2, 1, window=2, hashes=("1", "2"))
        overlap = self.linear(3, 1, window=2, hashes=("2", "3"))
        head = self.linear(4, 2, window=2, hashes=("3", "4"))
        self.history.commit_linear([left, overlap])
        self.history.commit_linear([head])
        self.assertEqual([self.history.linear_state(k) for k in (left, overlap, head)],
                         ["superseded", "live", "live"])

    def test_linear_drop_cost_uses_latest_joint_rollback_boundary(self):
        full = [HybridKey(str(n)) for n in range(1, 5)]
        old = self.linear(2, 1)
        current = self.linear(4, 2)
        self.history.bind(full)
        self.history.commit_linear([old, current])
        cases = (({old: 1, full[0]: 1}, full[1:], 2),
                 ({old: 1, full[0]: 1}, full[1:2], 2),
                 ({old: 1}, full[1:], 4))
        for disk, resident_full, expected_cost in cases:
            with self.subTest(disk=disk, resident_full=resident_full):
                dram = {key: 1 for key in [*resident_full, current]}
                decision = choose(dram, {current}, disk, set(), 8, self.history, 1,
                                  lambda _: 10, 1)
                self.assertEqual(decision.key, current)
                self.assertEqual(decision.drop_cost, expected_cost)

    def test_reuses_replacement_plan_for_equal_space_and_protection(self):
        full = [HybridKey(str(n), f"session-{n}") for n in range(4)]
        linear = [self.linear(n, 1, session=f"other-{n}") for n in range(1, 5)]
        victim = HybridKey("victim")
        for key in [*full, victim]:
            self.history.bind([key])
        self.history.commit_linear(linear)
        dram = dict.fromkeys([*full, *linear], 1)
        with patch("bidaw.gandalf_policy.plan_victims", return_value=((victim,), 1)) as plan:
            choose(dram, set(dram), {victim: 1}, set(), 1, self.history, 1, lambda _: .01, 1)
        self.assertEqual(plan.call_count, 1)


class NativeEvictionEligibilityTests(unittest.TestCase):
    def test_resident_candidates_match_native_read_and_staging_checks(self):
        from lmcache.v1.distributed.l1_manager.dram import DramL1Manager

        manager = object.__new__(DramL1Manager)
        manager._lock = Lock()
        key = Key("resident")
        for read_locked in (False, True):
            for staging_locks in (None, (), (False,), (True,), (True, False)):
                with self.subTest(read_locked=read_locked, staging_locks=staging_locks):
                    manager._objects = {key: SimpleNamespace(
                        read_lock=SimpleNamespace(is_locked=lambda: read_locked),
                        memory_obj=SimpleNamespace(get_size=lambda: 2),
                    )}
                    manager._staging = {} if staging_locks is None else {
                        key: {
                            tag: SimpleNamespace(write_lock=SimpleNamespace(
                                is_locked=lambda locked=locked: locked,
                            ))
                            for tag, locked in enumerate(staging_locks)
                        },
                    }
                    dram, eligible, _ = _l1_eviction_snapshot(manager, include_staging=False)
                    self.assertEqual(dram, {key: 2})
                    self.assertEqual(key in eligible, manager.is_key_evictable(key))


class FakeAdapter:
    _use_odirect = False
    _read_ahead_size = None
    _os_disk_bs = 512

    def __init__(self, directory):
        self.directory = Path(directory)
        self.writes = []
        self.deleted = []
        self.accessed = []
        self._lock = Lock()
        self._completed_load_tasks = {}
        self._load_efd = Mock()

    def _key_to_file_and_tmp_path(self, key):
        return self.directory / key.name, self.directory / (key.name + ".tmp")

    def _key_to_path(self, key):
        return self.directory / key.name

    async def _key_exists_on_disk(self, key):
        return self._key_to_file_and_tmp_path(key)[0].is_file()

    def _notify_keys_stored(self, keys, sizes):
        self.writes.extend(zip(keys, sizes, strict=True))

    def _notify_keys_deleted(self, keys, sizes):
        self.deleted.extend(zip(keys, sizes, strict=True))

    def _notify_keys_accessed(self, keys):
        self.accessed.extend(keys)

    async def delete_files(self, keys):
        removed, sizes = [], []
        for key in keys:
            path, _ = self._key_to_file_and_tmp_path(key)
            if path.is_file():
                sizes.append(path.stat().st_size)
                path.unlink()
                removed.append(key)
        return removed, sizes


class BoundedFilesystemTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.adapter = FakeAdapter(temporary.name)
        settings = Settings("gandalf", 4, 1, 1, .01, 100, 0, 8)
        self.budget = BoundedFS(self.adapter, settings, FakeAdapter.delete_files)
        self.adapter._gandalf_budget = self.budget
        self.addCleanup(self.budget._write_executor.shutdown)
        self.addCleanup(self.budget._read_executor.shutdown)
        self.a, self.b, self.c = Key("a"), Key("b"), Key("c")

    async def test_batch_reads_concurrently_and_reports_partial_success_in_input_order(self):
        missing, failed, short = (Key(name) for name in ("missing", "failed", "short"))
        keys = [self.a, missing, failed, short, self.b]
        objects = [SimpleNamespace(byte_array=bytearray(2)) for _ in keys]
        for key, payload in ((self.a, b"aa"), (self.b, b"bb"), (short, b"s")):
            self.adapter._key_to_path(key).write_bytes(payload)
        loop = asyncio.get_running_loop()
        started = asyncio.Queue()
        finish = Event()

        def read(adapter, key, obj):
            if key == failed:
                raise OSError(errno.EIO, "test I/O failure")
            if key in (self.a, self.b):
                loop.call_soon_threadsafe(started.put_nowait, (key, current_thread().name))
                if not finish.wait(5):
                    raise TimeoutError("concurrent read did not start")
            return _read_one_object(adapter, key, obj)

        with (patch("bidaw.gandalf_runtime._read_one_object", side_effect=read),
              self.assertLogs("bidaw.gandalf_runtime", level="WARNING")):
            batch = asyncio.create_task(self.budget.execute_load(keys, objects, 7))
            try:
                workers = [await asyncio.wait_for(started.get(), 2) for _ in range(2)]
                self.assertCountEqual([key for key, _ in workers], [self.a, self.b])
                self.assertTrue(all(name.startswith("gandalf-ssd-read") for _, name in workers))
            finally:
                finish.set()
                self.assertIsNone(await asyncio.wait_for(batch, 2))
        self.assertEqual(self.adapter._completed_load_tasks[7].get_indices_list(), [0, 4])
        self.assertEqual(self.adapter.accessed, [self.a, self.b])
        self.adapter._load_efd.notify.assert_called_once_with()
        self.assertEqual([objects[i].byte_array for i in (0, 4)], [b"aa", b"bb"])
        self.assertFalse(self.budget.readers)

    async def test_cancelled_load_retains_destination_and_ssd_pins_until_worker_exits(self):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finish = Event()
        freed = []
        memory_manager = SimpleNamespace(free=freed.extend)
        self.budget.source_pins = SourcePins(memory_manager, "ok")
        obj = SimpleNamespace(byte_array=bytearray(2), is_valid=lambda: True)

        def read(adapter, key, destination):
            loop.call_soon_threadsafe(started.set)
            if not finish.wait(5):
                raise TimeoutError("read not released")
            destination.byte_array[:] = b"ok"
            return 2, 2, False

        with patch("bidaw.gandalf_runtime._read_one_object", side_effect=read):
            batch = asyncio.create_task(self.budget.execute_load([self.a], [obj], 7))
            try:
                await asyncio.wait_for(started.wait(), 2)
                batch.cancel()
                await asyncio.sleep(0)
                self.assertFalse(batch.done())
                self.assertEqual(self.budget.readers[self.a], 1)
                self.assertEqual(memory_manager.free([obj]), "ok")
                self.assertFalse(freed)
            finally:
                finish.set()
                await asyncio.gather(batch, return_exceptions=True)
        self.assertTrue(batch.cancelled())
        self.assertEqual(obj.byte_array, b"ok")
        self.assertEqual(freed, [obj])
        self.assertFalse(self.budget.readers)
        self.assertFalse(self.adapter._completed_load_tasks)

    def test_odirect_counts_aligned_reads_and_writes_and_unaligned_fallbacks(self):
        self.adapter._use_odirect = True
        native_open = os.open
        direct_calls = []

        def open_file(path, flags, *args):
            if flags & os.O_DIRECT:
                direct_calls.append(path)
            return native_open(path, flags & ~os.O_DIRECT, *args)

        with (mmap.mmap(-1, 8192) as mapping, memoryview(mapping) as view,
              view[:4096] as aligned, view[1:4097] as unaligned,
              patch("bidaw.gandalf_runtime.os.open", side_effect=open_file)):
            aligned[:] = b"a" * 4096
            self.budget._write_file(self.a, aligned)
            self.assertEqual(_read_one_object(self.adapter, self.a, SimpleNamespace(byte_array=aligned)),
                             (4096, 4096, True))
            unaligned[:] = b"b" * 4096
            self.budget._write_file(self.b, unaligned)
            self.assertEqual(_read_one_object(self.adapter, self.b, SimpleNamespace(byte_array=unaligned)),
                             (4096, 4096, False))
        self.assertEqual(len(direct_calls), 2)
        for name in ("odirect_reads", "odirect_read_fallbacks", "odirect_writes", "odirect_write_fallbacks"):
            self.assertEqual(self.budget.status()[name], 1)

    def test_odirect_einval_retries_in_buffered_mode(self):
        self.adapter._use_odirect = True
        native_open = os.open

        def open_file(path, flags, *args):
            if flags & os.O_DIRECT:
                raise OSError(errno.EINVAL, "direct I/O unavailable")
            return native_open(path, flags, *args)

        with (mmap.mmap(-1, 4096) as mapping, memoryview(mapping) as view,
              patch("bidaw.gandalf_runtime.os.open", side_effect=open_file)):
            view[:] = b"a" * 4096
            self.budget._write_file(self.a, view)
            self.assertEqual(_read_one_object(self.adapter, self.a, SimpleNamespace(byte_array=view)),
                             (4096, 4096, False))
            self.assertEqual(bytes(view), b"a" * 4096)
        self.assertEqual(self.budget.status()["odirect_write_fallbacks"], 1)
        self.assertEqual(self.budget.status()["odirect_read_fallbacks"], 1)
        self.assertEqual(self.budget.status()["odirect_reads"], 0)

    async def test_load_only_waits_for_an_unlink_of_its_own_key(self):
        await self.budget.admit(self.a, b"aa")
        await self.budget.admit(self.b, b"bb")
        unlink_started = asyncio.Event()
        finish_unlink = asyncio.Event()
        conflict_started = asyncio.Event()
        load_calls = []

        async def unlink(adapter, keys):
            self.assertEqual(keys, [self.a])
            self.assertTrue(self.budget.io_lock.locked())
            self.assertIn(self.a, self.budget._removing)
            unlink_started.set()
            await finish_unlink.wait()
            return await FakeAdapter.delete_files(adapter, keys)

        async def load(adapter, keys, objects, task_id):
            load_calls.append(task_id)
            self.assertTrue(self.budget.readers[keys[0]])
            return await adapter._key_exists_on_disk(keys[0])

        async def conflicting_load():
            conflict_started.set()
            return await self.budget.execute_load([self.a], [], "conflict", load)

        with patch.object(self.budget, "delete_files", side_effect=unlink):
            store = asyncio.create_task(self.budget.admit(self.c, b"cc", (self.a,)))
            conflict = None
            try:
                await asyncio.wait_for(unlink_started.wait(), 2)
                conflict = asyncio.create_task(conflicting_load())
                await asyncio.wait_for(conflict_started.wait(), 2)
                self.assertTrue(await asyncio.wait_for(
                    self.budget.execute_load([self.b], [], "independent", load), 2,
                ))
                self.assertEqual(load_calls, ["independent"])
                self.assertFalse(conflict.done())
                self.assertNotIn(self.a, self.budget.readers)
            finally:
                finish_unlink.set()
                await asyncio.wait_for(store, 2)
                if conflict is not None:
                    self.assertFalse(await asyncio.wait_for(conflict, 2))
        self.assertEqual(load_calls, ["independent", "conflict"])
        self.assertFalse(self.budget.readers)

    async def test_store_notifies_after_committing_and_uses_the_write_executor(self):
        write = self.budget._write_file
        threads = []

        def tracked_write(key, data):
            threads.append(current_thread().name)
            write(key, data)

        def inspect_commit(keys, sizes):
            self.assertTrue(self.budget.lock.acquire(blocking=False))
            self.budget.lock.release()
            self.assertEqual(keys, [self.a])
            self.assertEqual(sizes, [2])
            self.assertEqual(self.budget.residents, {self.a: 2})
            self.assertEqual(self.budget.inflight_keys, {})
            self.assertEqual(self.budget.status(), {
                "writes": 1, "write_bytes": 2, "used_bytes": 2,
                "resident_objects": 1, "inflight_bytes": 0,
                "odirect_reads": 0, "odirect_read_fallbacks": 0,
                "odirect_writes": 0, "odirect_write_fallbacks": 0,
            })

        with (
            patch.object(self.budget, "_write_file", side_effect=tracked_write),
            patch.object(self.adapter, "_notify_keys_stored", side_effect=inspect_commit) as notify,
        ):
            self.assertEqual(await self.budget.admit(self.a, b"aa"), (Admission.STORED, 2))
        notify.assert_called_once()
        self.assertEqual(len(threads), 1)
        self.assertTrue(threads[0].startswith("gandalf-ssd-write"))

    async def test_duplicate_concurrent_write_counts_once_and_respects_capacity(self):
        results = await asyncio.gather(*(self.budget.admit(self.a, b"aa") for _ in range(2)))
        self.assertCountEqual(results, [(Admission.STORED, 2), (Admission.STORED, 0)])
        await self.budget.admit(self.b, b"bb")
        self.budget.touch([self.a])
        await self.budget.admit(self.c, b"cc")
        self.assertEqual(self.adapter.deleted, [(self.b, 2)])
        self.assertEqual(self.budget.status()["used_bytes"], 4)
        self.assertEqual(self.budget.status()["write_bytes"], 6)
        self.assertTrue(await self.adapter._key_exists_on_disk(self.c))

    async def test_in_flight_writes_reserve_quota_and_run_concurrently(self):
        loop = asyncio.get_running_loop()
        started = asyncio.Queue()
        release = Event()
        write = self.budget._write_file

        def wait_and_write(key, data):
            loop.call_soon_threadsafe(started.put_nowait, key)
            if release.wait(3):
                write(key, data)

        with patch.object(self.budget, "_write_file", side_effect=wait_and_write):
            writes = [asyncio.create_task(self.budget.admit(key, b"ab", ()))
                      for key in (self.a, self.b)]
            try:
                running = [await asyncio.wait_for(started.get(), 2) for _ in writes]
                self.assertCountEqual(running, [self.a, self.b])
                self.assertEqual(self.budget.status()["inflight_bytes"], 4)
                self.assertEqual(self.budget.status()["used_bytes"], 0)
                self.assertEqual(await self.budget.admit(self.c, b"ab", ()),
                                 (Admission.BLOCKED, 0))
            finally:
                release.set()
                results = await asyncio.gather(*writes)
        self.assertEqual(results, [(Admission.STORED, 2), (Admission.STORED, 2)])
        self.assertEqual(self.budget.status()["used_bytes"], 4)
        self.assertEqual(self.budget.status()["inflight_bytes"], 0)
        self.assertEqual(self.budget.inflight_keys, {})

    async def test_cancelled_admission_keeps_memoryview_pinned_until_writer_finishes(self):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = Event()
        write = self.budget._write_file

        def wait_and_write(key, data):
            loop.call_soon_threadsafe(started.set)
            if release.wait(3):
                write(key, data)

        freed = []
        memory_manager = SimpleNamespace(free=freed.extend)
        self.budget.source_pins = SourcePins(memory_manager, "ok")
        obj = SimpleNamespace(byte_array=bytearray(b"ab"), is_valid=lambda: True)
        with patch.object(self.budget, "_write_file", side_effect=wait_and_write):
            task = asyncio.create_task(self.budget.write_object(self.a, obj))
            try:
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertEqual(memory_manager.free([obj]), "ok")
                self.assertEqual(freed, [])
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertEqual(freed, [obj])
        self.assertEqual(self.budget.status()["inflight_bytes"], 0)
        self.assertEqual(self.adapter.writes, [(self.a, 2)])
        self.assertEqual(self.adapter._key_to_file_and_tmp_path(self.a)[0].read_bytes(), b"ab")

    async def test_pinned_victim_blocks_and_io_failure_does_not_count_as_write(self):
        await self.budget.admit(self.a, b"aaaa")
        await self.budget.pin_load([self.a])
        self.assertEqual(await self.budget.admit(self.b, b"bb"), (Admission.BLOCKED, 0))
        self.budget.unpin_load([self.a])
        self.assertTrue(await self.adapter._key_exists_on_disk(self.a))
        with patch.object(self.budget, "_write_file", side_effect=OSError("full disk")):
            with self.assertLogs("bidaw.gandalf_runtime", level="ERROR"):
                result = await self.budget.admit(self.b, b"bb", (self.a,))
        self.assertEqual(result, (Admission.FAILED, 0))
        self.assertFalse(await self.adapter._key_exists_on_disk(self.b))
        self.assertEqual(self.budget.status()["writes"], 1)
        self.assertEqual(self.budget.status()["used_bytes"], 0)

    async def test_expired_migration_at_admission_does_not_evict_or_write(self):
        await self.budget.admit(self.a, b"aaaa")
        result = await self.budget.admit(self.b, b"bb", (self.a,), can_move=lambda: False)
        self.assertEqual(result, (Admission.BLOCKED, 0))
        self.assertTrue(await self.adapter._key_exists_on_disk(self.a))
        self.assertFalse(await self.adapter._key_exists_on_disk(self.b))

    async def test_l1_release_follows_success_and_failed_migration_retains_copy(self):
        self.adapter._loop = asyncio.get_running_loop()
        memory = SimpleNamespace(byte_array=b"aa")
        lifecycle = []

        def reserve(keys):
            lifecycle.append("reserve")
            return {self.a: ("ok", memory)}

        def finish(keys):
            lifecycle.append("unlock")

        def delete(keys):
            self.assertTrue(self.adapter._key_to_file_and_tmp_path(self.a)[0].is_file())
            lifecycle.append("delete")
            return {self.a: "ok"}

        manager = SimpleNamespace(reserve_read=reserve, finish_read=finish, delete=delete)
        decision = Decision(self.a, "move")
        with patch.object(self.budget, "_write_file", side_effect=OSError("disk full")):
            with self.assertLogs("bidaw.gandalf_runtime", level="ERROR"):
                await asyncio.to_thread(reclaim, manager, self.budget, decision, "ok")
        self.assertEqual(lifecycle, ["reserve", "unlock"])
        self.assertEqual(self.budget.status()["failed_migrations"], 1)

        await asyncio.to_thread(reclaim, manager, self.budget, decision, "ok")
        self.assertEqual(lifecycle, ["reserve", "unlock", "reserve", "unlock", "delete"])
        self.assertEqual(self.budget.status()["migrations"], 1)


if __name__ == "__main__":
    unittest.main()
