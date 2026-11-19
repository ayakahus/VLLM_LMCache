import unittest

from bidaw.l2_fate_report import classify


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

    def test_all_later_lookups_must_hit_l1_and_never_attempt_ssd(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.start(1)
        self.event("lookup", request="1")
        self.event("l1_hit", request="1")
        self.finish(1, last=True)
        self.assertEqual(self.counts(), dict(A=1, B=0, C=0, U=0))

        self.events.pop()
        self.events.pop()
        self.event("lookup", request="2")
        self.finish(1, last=True)
        self.assertEqual(self.counts()["U"], 1)

        self.events.pop()
        self.events.pop()
        self.event("l1_hit", request="2")
        self.event("load_attempt", request="2", adapter="ssd")
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
        self.assertEqual(self.counts()["B"], 1)

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
        self.assertEqual(self.counts()["B"], 1)
        self.events.pop()
        marker("start", "reader", 0)
        event("lookup", "", request="reuse", round_id="reader")
        marker("finish", "reader", 0)
        marker("session_end", "reader", 0)
        event("shared_replay_end", "")
        self.assertEqual(self.counts()["U"], 1)

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
            ("lookup", {"request": "later"}),
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

        self.event("load_attempt", adapter="ssd", io_task="1")
        self.event("load", adapter="ssd", io_task="1")
        report = classify(self.events)
        self.assertEqual(report["classes"]["U"], 1)
        self.assertEqual(report["load_events"], 1)
        self.assertEqual(report["attributed_load_events"], 1)

    def readback(self, *, lookup_round_id=None, removed_again=False, padded=False):
        self.start(0)
        if padded:
            self.event("write", adapter="ssd", bytes=7340032, tail_valid_tokens=88)
        else:
            self.write()
        self.finish(0)
        self.event("l1_evict", policy="bidaw_eviction_policy")
        if removed_again:
            self.event("l1_remove")
        self.start(1)
        self.event("lookup", request="1", round_id=lookup_round_id or self.round_id)
        self.event("preparing", request="1")
        self.event("load_attempt", request="1", adapter="ssd")
        self.event("load", request="1", adapter="ssd")

    def test_effective_write_requires_linked_preparing_dispatch_and_l2_hit(self):
        self.readback()
        self.assertEqual(self.counts()["U"], 1)
        self.event("l2_hit", request="other")
        self.assertEqual(self.counts()["U"], 1)
        self.event("l2_hit", request="1")
        self.assertEqual(self.counts()["U"], 1)
        self.finish(1)
        self.assertEqual(self.counts()["C"], 1)
        self.assertEqual(classify(self.events)["qualifying_prefetch_write_pairs"], 1)

        self.event("delete", adapter="ssd", eviction=False)
        self.write(size=11)
        result = classify(self.events)
        self.assertEqual(result["classes"], dict(A=0, B=0, C=1, U=1))
        self.assertEqual(result["bytes"]["U"], 11)
        self.assertEqual(result["rewrite_events"], 1)
        self.events[3]["policy"] = "LRU"
        self.assertEqual(self.counts()["U"], 2)

    def test_c_requires_eviction_before_lookup_and_accepts_eviction_during_round(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.start(1)
        self.event("l1_evict", policy="bidaw_eviction_policy")
        eviction = self.events[-1]
        self.event("lookup", request="1")
        lookup = self.events[-1]
        self.event("preparing", request="1")
        self.event("load_attempt", request="1", adapter="ssd")
        self.event("load", request="1", adapter="ssd")
        self.event("l2_hit", request="1")
        self.finish(1)
        self.assertEqual(self.counts()["C"], 1)

        eviction["t_ns"], lookup["t_ns"] = lookup["t_ns"], eviction["t_ns"]
        self.assertEqual(self.counts()["U"], 1)

    def test_same_salt_readback_without_matching_round_is_not_c(self):
        self.readback(lookup_round_id="another-request")
        self.event("l2_hit", request="1")
        self.finish(1)
        report = classify(self.events)
        self.assertEqual(report["writes_read_back"], 1)
        self.assertEqual(report["classes"], dict(A=0, B=0, C=0, U=1))

    def test_later_non_policy_removal_invalidates_old_bidaw_eviction(self):
        self.readback(removed_again=True)
        self.event("l2_hit", request="1")
        self.finish(1)
        self.assertEqual(self.counts(), dict(A=0, B=0, C=0, U=1))

    def test_padded_write_is_not_c_in_normal_replay(self):
        self.readback(padded=True)
        self.event("l2_hit", request="1")
        self.finish(1)
        self.assertEqual(self.counts(), dict(A=0, B=0, C=0, U=1))

    def test_legacy_log_cannot_claim_a_b_or_c_from_timestamps_alone(self):
        self.readback()
        self.event("l2_hit", request="1")
        self.finish(1, last=True)
        for event in self.events:
            event.pop("round_id")
        report = classify(self.events)
        self.assertEqual(report["writes_read_back"], 1)
        self.assertEqual(report["writes_without_round_evidence"], 1)
        self.assertEqual(report["classes"], dict(A=0, B=0, C=0, U=1))

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
        self.assertEqual(report["classes"], dict(A=0, B=0, C=0, U=1))
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
