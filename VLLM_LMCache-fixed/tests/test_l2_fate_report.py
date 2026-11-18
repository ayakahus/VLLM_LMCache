import unittest

from bidaw.l2_fate_report import classify


class ChunkFateTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.key = dict(model="m", rank=0, group=0, salt="u", hash="k")

    def event(self, kind, **fields):
        self.events.append(dict(self.key, t_ns=len(self.events), ev=kind, **fields))

    def write(self, size=7):
        self.event("write", adapter="ssd", bytes=size)

    def start(self, number):
        self.event("round_start", round=number)

    def finish(self, number, *, last=False):
        self.event("round_finish", round=number)
        if last:
            self.event("round_session_end", round=number)

    def counts(self):
        return classify(self.events)["classes"]

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

    def test_dead_write_requires_session_end_and_both_evictions(self):
        self.start(0)
        self.write()
        self.event("l1_evict", policy="bidaw_eviction_policy")
        self.finish(0)
        self.event("delete", adapter="ssd", eviction=True)
        self.assertEqual(self.counts()["U"], 1)
        self.event("round_session_end", round=0)
        self.assertEqual(self.counts()["B"], 1)
        self.events[-2]["eviction"] = False
        self.assertEqual(self.counts()["U"], 1)

    def test_effective_write_requires_linked_preparing_dispatch_and_l2_hit(self):
        self.start(0)
        self.write()
        self.finish(0)
        self.event("l1_evict", policy="bidaw_eviction_policy")
        self.start(1)
        self.event("lookup", request="1")
        self.event("preparing", request="1")
        self.event("load_attempt", request="1", adapter="ssd")
        self.event("load", request="1", adapter="ssd")
        self.assertEqual(self.counts()["U"], 1)
        self.event("l2_hit", request="other")
        self.assertEqual(self.counts()["U"], 1)
        self.event("l2_hit", request="1")
        self.assertEqual(self.counts()["C"], 1)

        self.event("delete", adapter="ssd", eviction=False)
        self.write(size=11)
        result = classify(self.events)
        self.assertEqual(result["classes"], dict(A=0, B=0, C=1, U=1))
        self.assertEqual(result["bytes"]["U"], 11)
        self.assertEqual(result["rewrite_events"], 1)
        self.events[3]["policy"] = "LRU"
        self.assertEqual(self.counts()["U"], 2)
