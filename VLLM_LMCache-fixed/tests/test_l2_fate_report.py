import unittest

from bidaw.l2_fate_report import LABELS, classify


class ChunkFateTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.key = dict(model="m", rank=0, group=0, salt="u", hash="k")
        self.round_id = None

    def event(self, kind, **fields):
        self.events.append({
            **self.key, "t_ns": len(self.events), "ev": kind,
            "round_id": self.round_id, **fields,
        })

    def write(self, size=7):
        self.event("write", adapter="ssd", bytes=size)

    def start(self, number):
        self.round_id = f"{self.key['salt']}-{number}"
        self.event("round_start", round=number)

    def finish(self, number, *, last=False):
        self.event("round_finish", round=number)
        if last:
            self.event("round_session_end", round=number)

    def counts(self):
        return classify(self.events)["classes"]

    def test_padded_tail_is_included_once_at_full_physical_size(self):
        self.start(0)
        self.write(size=7340032)
        self.key = dict(self.key, hash="padded")
        self.event("write", adapter="ssd", bytes=7340032, tail_valid_tokens=88)
        self.finish(0, last=True)
        report = classify(self.events)
        self.assertEqual(report["physical_writes"], 2)
        self.assertEqual(report["padded_tail_writes"], 1)
        self.assertEqual(sum(report["bytes"].values()), 2 * 7340032)

    def test_all_later_lookups_hit_dram_after_ssd_eviction_without_session_end(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.event("ssd_eviction", bytes=7)
        self.start(1)
        self.event("lookup", request="1")
        self.event("l1_hit", request="1")
        self.event("load_attempt", request="1", adapter="ssd")
        self.finish(1)
        self.assertEqual(self.counts()["A"], 1)

    def test_first_dram_miss_uses_copy_residency_at_lookup_time(self):
        for removal, expected in (("before", "B2"), ("after", "B3"), ("absent", "B3")):
            with self.subTest(removal=removal):
                self.events.clear()
                self.start(0)
                self.write()
                self.finish(0)
                self.start(1)
                self.event("lookup", request="hit")
                self.event("l1_hit", request="hit")
                if removal == "before":
                    self.event("ssd_eviction", bytes=7)
                self.event("lookup", request="first-miss")
                if removal == "after":
                    self.event("delete", adapter="ssd", eviction=True)
                self.event("lookup", request="second-miss")
                self.finish(1)
                self.assertEqual(self.counts()["U"], 1)
                self.event("round_session_end", round=1)
                self.assertEqual(self.counts()[expected], 1)

    def test_l2_hit_missing_readback_evidence_is_unresolved(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.start(1)
        self.event("lookup", request="1")
        self.event("l2_hit", request="1")
        self.finish(1, last=True)
        self.assertEqual(self.counts()["U"], 1)

    def test_unused_write_requires_session_end_but_not_eviction_or_final_round(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.start(1)
        self.finish(1)
        self.assertEqual(self.counts()["U"], 1)
        self.event("round_session_end", round=1)
        self.assertEqual(self.counts()["B1"], 1)

    def test_store_draining_after_a_completed_session_is_natural_dead_write(self):
        self.start(0)
        self.finish(0, last=True)
        self.write()
        self.assertEqual(self.counts()["B1"], 1)

    def test_shared_write_checks_other_sessions_and_waits_for_replay_end(self):
        def event(kind, salt, **fields):
            self.events.append({**self.key, "ev": kind, "salt": salt,
                                "t_ns": len(self.events), **fields})

        def marker(phase, session, number):
            event(f"round_{phase}", session, round=number, round_id=session, shared_cache=True)

        marker("start", "writer", 2)
        event("write", "", adapter="ssd", bytes=7, round_id="writer")
        marker("finish", "writer", 2)
        marker("session_end", "writer", 2)
        self.assertEqual(self.counts()["U"], 1)

        event("shared_replay_end", "")
        self.assertEqual(self.counts()["B1"], 1)
        self.events.pop()
        marker("start", "reader", 0)
        event("lookup", "", request="reuse", round_id="reader")
        marker("finish", "reader", 0)
        marker("session_end", "reader", 0)
        event("shared_replay_end", "")
        self.assertEqual(self.counts()["B3"], 1)

        event("l1_hit", "", request="reuse")
        self.assertEqual(self.counts()["A"], 1)

    def test_lookup_or_read_after_copy_closes_rules_out_b(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.event("delete", adapter="ssd", eviction=False)
        self.start(1)
        before_access = len(self.events)
        for kind, fields in (
            ("lookup", {"request": "unscoped", "round_id": "missing"}),
            ("load_attempt", {"adapter": "ssd", "io_task": "later"}),
        ):
            with self.subTest(kind=kind, fields=fields):
                del self.events[before_access:]
                self.event(kind, **fields)
                self.finish(1, last=True)
                self.assertEqual(self.counts()["U"], 1)

    def test_legacy_recency_notification_does_not_count_as_an_ssd_read(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.start(1)
        self.event("lookup", request="1")
        self.event("l1_hit", request="1")
        self.event("load", adapter="ssd")
        self.finish(1, last=True)

        report = classify(self.events)
        self.assertEqual(report["classes"]["A"], 1)
        self.assertEqual(report["load_events"], 0)
        self.assertEqual(report["unattributed_load_events"], 0)
        self.assertEqual(report["legacy_touch_events"], 1)

        del self.events[-2:]
        self.event("load_attempt", adapter="ssd", io_task="1")
        self.event("load", adapter="ssd", io_task="1")
        self.finish(1, last=True)
        report = classify(self.events)
        self.assertEqual(report["classes"]["U"], 1)
        self.assertEqual(report["load_events"], 1)
        self.assertEqual(report["attributed_load_events"], 1)

    def readback(self, *, padded=False):
        self.start(0)
        if padded:
            self.event("write", adapter="ssd", bytes=7340032, tail_valid_tokens=88)
        else:
            self.write()
        self.finish(0)
        self.start(1)
        self.event("lookup", request="1")
        self.event("load_attempt", request="1", adapter="ssd", io_task="read")
        self.event("load", request="1", adapter="ssd", io_task="read")

    def test_c_precedes_a_and_an_incomplete_window(self):
        self.readback()
        self.event("l1_hit", request="1")
        self.event("l2_hit", request="1")
        self.assertEqual(self.counts()["C"], 1)

    def test_ssd_hit_is_attributed_only_to_the_rewritten_copy(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.event("delete", adapter="ssd", eviction=True)
        self.start(1)
        self.write(size=11)
        self.event("lookup", request="1")
        self.event("load_attempt", request="1", adapter="ssd")
        self.event("load", request="1", adapter="ssd")
        self.event("l2_hit", request="1")
        self.finish(1, last=True)
        result = classify(self.events)
        self.assertEqual(result["classes"], {**dict.fromkeys(LABELS, 0), "B2": 1, "C": 1})
        self.assertEqual(result["bytes"]["C"], 11)
        self.assertEqual(result["rewrite_events"], 1)

    def test_confirmed_readback_c_matches_io_and_hit_without_policy_events(self):
        self.write()
        self.event("l2_hit", request="r")  # Too early to confirm this read.
        self.event("load_attempt", request="r", adapter="ssd", io_task="read")
        self.event("load", request="r", adapter="ssd", io_task="read")
        self.event("l2_hit", request="other")
        self.assertEqual(classify(self.events)["classes"]["U"], 1)

        self.event("l2_hit", request="r")
        result = classify(self.events)
        self.assertEqual(result["classes"], {**dict.fromkeys(LABELS, 0), "C": 1})
        self.assertEqual(result["bytes"]["C"], 7)
        self.assertEqual(result["qualifying_prefetch_write_pairs"], 1)

    def test_confirmed_readback_does_not_assign_a_hit_to_ambiguous_copies(self):
        for adapter in ("ssd-1", "ssd-2"):
            self.event("write", adapter=adapter, bytes=7)
            self.event("load_attempt", request="r", adapter=adapter, io_task=adapter)
            self.event("load", request="r", adapter=adapter, io_task=adapter)
        self.event("l2_hit", request="r")
        result = classify(self.events)
        self.assertEqual(result["writes_read_back"], 2)
        self.assertEqual(result["classes"], {**dict.fromkeys(LABELS, 0), "U": 2})

    def test_confirmed_padded_write_counts_as_c(self):
        self.readback(padded=True)
        self.event("l2_hit", request="1")
        self.finish(1)
        self.assertEqual(self.counts()["C"], 1)

    def test_legacy_log_cannot_infer_b_or_a_from_timestamps_alone(self):
        self.readback()
        self.finish(1, last=True)
        for event in self.events:
            event.pop("round_id")
        report = classify(self.events)
        self.assertEqual(report["writes_read_back"], 1)
        self.assertEqual(report["writes_without_round_evidence"], 1)
        self.assertEqual(report["classes"]["U"], 1)

    def test_load_frequency_counts_only_matched_write_lifecycles(self):
        self.start(0)
        self.event("load_attempt", request="old", adapter="ssd", io_task="0")
        self.event("load", request="old", adapter="ssd", io_task="0")
        self.write()
        for task in ("1", "2"):
            self.event("load_attempt", request="same-prefetch", adapter="ssd", io_task=task)
            self.event("load", request="same-prefetch", adapter="ssd", io_task=task)
        self.finish(0, last=True)

        report = classify(self.events)
        self.assertEqual(report["classes"], {**dict.fromkeys(LABELS, 0), "U": 1})
        self.assertEqual(report["load_events"], 3)
        self.assertEqual(report["attributed_load_events"], 2)
        self.assertEqual(report["unattributed_load_events"], 1)
        self.assertEqual(report["writes_read_back"], 1)
        self.assertEqual(report["readback_prefetch_write_pairs"], 1)
        self.assertEqual(report["qualifying_prefetch_write_pairs"], 0)

    def test_load_finishing_after_rewrite_is_unattributed(self):
        self.start(0)
        self.write()
        self.event("load_attempt", request="r", adapter="ssd", io_task="before")
        self.event("delete", adapter="ssd", eviction=False)
        self.write()
        self.event("load_attempt", request="r", adapter="ssd", io_task="after")
        self.event("load", request="r", adapter="ssd", io_task="before")

        report = classify(self.events)
        self.assertEqual(report["load_events"], 1)
        self.assertEqual(report["unattributed_load_events"], 1)
        self.assertEqual(report["writes_read_back"], 0)
        self.event("l2_hit", request="r")
        self.assertEqual(classify(self.events)["classes"]["C"], 0)
