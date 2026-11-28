from collections import defaultdict
import unittest

from bidaw.ttft_breakdown import record_ssd_and_prefetch, stats


def batch_events(task, size, start_ms, wall_ms, *, detailed):
    start, end = start_ms * 1_000_000, (start_ms + wall_ms) * 1_000_000
    fields = dict(io_task=task, adapter="local", request=task)
    events = [dict(ev="lookup", t_ns=start - 2_000_000, request=task)]
    for index in range(size):
        key = {**fields, "model": "model", "rank": 0, "group": 0,
               "salt": "session", "hash": str(index)}
        events.extend((dict(ev="load_attempt", t_ns=start, **key),
                       dict(ev="load", t_ns=end, **key)))
        if detailed:
            events.append(dict(ev="load_object_done", t_ns=end, **key,
                               io_ms=8, queue_ms=1, resume_ms=1, object_wall_ms=10))
    if detailed:
        events.append(dict(ev="load_task_done", t_ns=end, **fields,
                           task_wall_ms=wall_ms, wait_conflict_ms=0))
    return events


class SSDReadReportTests(unittest.TestCase):
    def collect(self, *, detailed):
        events = batch_events("first", 1, 10, 10, detailed=detailed)
        events.extend(batch_events("second", 9, 30, 90, detailed=detailed))
        samples = defaultdict(list)
        record_ssd_and_prefetch(sorted(events, key=lambda event: event["t_ns"]),
                                lambda _: True, samples, 2, has_io_ids=True)
        return samples

    def test_task_and_object_timings_use_different_sample_populations(self):
        samples = self.collect(detailed=True)
        self.assertEqual(stats(samples["ssd_load_task_ms"])["mean"], 50)
        self.assertEqual(samples["ssd_load_task_ms"], [10, 90])
        self.assertEqual(samples["ssd_object_io_ms"], [8] * 10)
        self.assertEqual(samples["ssd_object_overhead_ms"], [2] * 10)
        self.assertEqual(samples["ssd_object_wall_ms"], [10] * 10)
        self.assertEqual(samples["ssd_per_request"], [10, 90])
        self.assertEqual(samples["lookup_to_load_attempt_ms"], [2, 2])
        self.assertEqual(samples["ssd_read"], [])

    def test_legacy_event_pairs_reconstruct_tasks_and_keep_legacy_stat(self):
        samples = self.collect(detailed=False)
        self.assertEqual(stats(samples["ssd_load_task_ms"])["mean"], 50)
        self.assertEqual(stats(samples["ssd_read"])["mean"], 82)
        self.assertEqual(samples["ssd_object_io_ms"], [])
        self.assertEqual(samples["ssd_object_wall_ms"], [])
        self.assertEqual(samples["ssd_per_request"], [10, 90])
