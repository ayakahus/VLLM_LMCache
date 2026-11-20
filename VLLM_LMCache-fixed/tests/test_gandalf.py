import asyncio
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bidaw.gandalf_policy import Decision, History, choose
from bidaw.gandalf_runtime import Admission, BoundedFS, Settings, reclaim


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

    def test_replacement_plan_adds_costs_and_preserves_required_ancestors(self):
        victim, victim_child = Key("victim", "old"), Key("victim_child", "old")
        new, new_child, extra = Key("new", "incoming"), Key("new_child", "incoming"), Key("extra", "incoming")
        self.history.bind([victim, victim_child])
        self.history.bind([self.root, new, new_child, extra])
        dram = {new: 3, new_child: 1, extra: 1, self.root: 2}
        disk = {self.root: 1, victim: 2, victim_child: 1}
        decision = self.decision(dram, disk, eligible={new})
        self.assertEqual(decision.victims, (victim_child, victim))
        self.assertAlmostEqual(decision.replacement_cost, 1)
        # Equal total costs reject a demotion at the strict threshold.
        self.assertEqual(
            self.decision(dram, disk, eligible={new}, fee=1 / 6),
            None,
        )
        self.assertIsNone(self.decision(dram, disk, eligible={new}, pinned={victim_child}))

    def test_request_clock_deduplicates_retries(self):
        self.history.observe("session", "1")
        self.history.observe("other", "2")
        self.history.observe("session", "1")
        self.history.observe("session", "3")
        self.assertEqual(self.history.clock, 3)
        self.assertEqual(list(self.history.gaps), [2])


class FakeAdapter:
    _use_odirect = False

    def __init__(self, directory):
        self.directory = Path(directory)
        self.writes = []
        self.deleted = []

    def _key_to_file_and_tmp_path(self, key):
        return self.directory / key.name, self.directory / (key.name + ".tmp")

    async def _key_exists_on_disk(self, key):
        return self._key_to_file_and_tmp_path(key)[0].is_file()

    def _notify_keys_stored(self, keys, sizes):
        self.writes.extend(zip(keys, sizes, strict=True))

    def _notify_keys_deleted(self, keys, sizes):
        self.deleted.extend(zip(keys, sizes, strict=True))

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
        self.a, self.b, self.c = Key("a"), Key("b"), Key("c")

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

    async def test_pinned_victim_blocks_and_io_failure_does_not_count_as_write(self):
        await self.budget.admit(self.a, b"aaaa")
        self.budget.pin_load([self.a])
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
