"""CPU-only regression tests. LMCache ABI is mocked, not GPU-validated."""
import asyncio
import importlib
import logging
import os
import random
import subprocess
import sys
import threading
import time
import types
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bidaw import policy_registry
from bidaw.reuse_distance import ReuseDistance
from bidaw import trace_replay as replay


@dataclass(frozen=True)
class Identity:
    chunk_hash: bytes
    cache_salt: str = 'u'
    model_name: str = 'model'
    kv_rank: int = 0
    object_group_id: int = 0


@dataclass(frozen=True)
class Key:
    identity: Identity

    @property
    def cache_salt(self):
        return self.identity.cache_salt


def key(value, salt='u'):
    return Key(Identity(str(value).encode(), salt))


def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    return value


def lmcache_stubs():
    @dataclass
    class Action:
        destination: object
        keys: list
    return {
        'lmcache': module('lmcache'),
        'lmcache.logging': module('lmcache.logging', init_logger=logging.getLogger),
        'lmcache.v1': module('lmcache.v1'),
        'lmcache.v1.distributed': module('lmcache.v1.distributed'),
        'lmcache.v1.distributed.api': module('lmcache.v1.distributed.api', ObjectKey=Key),
        'lmcache.v1.distributed.eviction': module('lmcache.v1.distributed.eviction', EvictionPolicy=object),
        'lmcache.v1.distributed.internal_api': module('lmcache.v1.distributed.internal_api',
            EvictionAction=Action, EvictionDestination=types.SimpleNamespace(DISCARD='discard')),
    }


class DistanceTests(unittest.TestCase):
    def test_repeated_objects_and_adjacent_access(self):
        tracker = ReuseDistance()
        values = [tracker.access(k) for k in 'ABBAA']
        self.assertEqual(values, [None, None, 0, 1, 0])

    def test_matches_slow_distinct_reference_through_compaction(self):
        rng, trace, tracker = random.Random(12), [], ReuseDistance()
        for _ in range(12_000):
            k = rng.randrange(80)
            prev = next((i for i in range(len(trace)-1, -1, -1) if trace[i] == k), None)
            expected = None if prev is None else len(set(trace[prev+1:]))
            self.assertEqual(tracker.access(k), expected)
            trace.append(k)

    def test_bounded_history_censors_forgotten_identity(self):
        tracker = ReuseDistance(max_keys=2)
        for k in 'ABC':
            tracker.access(k)
        self.assertIsNone(tracker.access('A'))
        self.assertLessEqual(len(tracker.last), 2)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, lmcache_stubs()):
            cls.mod = importlib.import_module('bidaw.eviction_policy.bidaw_policy')

    def setUp(self):
        self.p = self.mod.BidawEvictionPolicy()
        self.p._ensure_ghost_thread = lambda: None

    def test_lifecycle_does_not_train_predictor(self):
        a = key('A')
        self.p.on_keys_created([a]); self.p.on_keys_touched([a])
        self.assertEqual(self.p._seq, 0)
        self.assertIn(a, self.p._tracked)

    def test_eviction_preserves_history(self):
        a, b = key('A'), key('B')
        self.p.on_keys_created([a])
        self.p.on_request_access([a]); self.p.on_keys_removed([a])
        self.p.on_request_access([b]); self.p.on_keys_created([a])
        self.p.on_request_access([a])
        self.assertEqual(list(self.p._salt_hist['u']), [1])

    def test_request_dedup_and_zero_distance(self):
        a = key('A')
        self.p.on_request_access([a, a]); self.p.on_request_access([a])
        self.assertEqual(self.p._seq, 2)
        self.assertEqual(list(self.p._salt_hist['u']), [0])

    def test_identity_preserves_tenant_and_rank(self):
        self.assertNotEqual(self.p._key_id(key('A', 'u')), self.p._key_id(key('A', 'v')))
        self.assertNotEqual(self.p._key_id(key('A')), self.p._key_id(Key(Identity(b'A', kv_rank=1))))

    def test_ghost_builds_and_records_snapshot_sequence(self):
        self.p._ghost_window = 256
        for i in range(200):
            self.p.on_request_access([key(i % 20)])
        self.p._rebuild_ghost()
        self.assertIsNotNone(self.p._ghost)
        self.assertEqual(self.p._ghost_built_seq, 200)
        self.assertEqual(self.p._ghost[0], 1.0)

    def test_ghost_matches_slow_belady_bucket_reference(self):
        self.p._ghost_window = 256
        self.p._capacity_keys = 3
        self.p._extreme_threshold = 18
        rng = random.Random(5)
        trace = [rng.randrange(10) for _ in range(180)]
        for value in trace:
            self.p.on_request_access([key(value)])
        self.p._rebuild_ghost()
        cache, buckets = set(), [[] for _ in range(self.p._promising_buckets)]
        for i, value in enumerate(trace):
            hit = value in cache
            prev = next((j for j in range(i-1,-1,-1) if trace[j] == value),None)
            if prev is not None:
                distance = len(set(trace[prev+1:i]))
                kind, idx = self.p._bucket_of(distance)
                if kind == 1: buckets[idx].append(int(hit))
            if not hit:
                if len(cache) >= 3:
                    def next_use(k):
                        return next((j for j in range(i+1,len(trace)) if trace[j] == k),float('inf'))
                    cache.remove(max(cache,key=next_use))
                cache.add(value)
        for i, bucket in enumerate(buckets):
            if bucket:
                self.assertAlmostEqual(self.p._ghost[1][i],sum(bucket)/len(bucket))

    def test_filter_protects_locked_keys_and_zero_ratio(self):
        a, b = key('A'), key('B')
        self.p.on_keys_created([a,b])
        self.assertEqual(self.p.get_eviction_actions(0), [])
        actions = self.p.get_eviction_actions(1, key_eligible_filter=lambda k: k != a)
        self.assertEqual([k for act in actions for k in act.keys], [b])

    def test_unknown_answer_does_not_train_lower_bound(self):
        for _ in range(12): self.p._record_wrd('u', 5)
        self.assertEqual(self.p._lb_min, {})
        self.p.on_answer('u', 64)
        for _ in range(12): self.p._record_wrd('u', 5)
        self.assertEqual(self.p._lb_min[2], 5)


class Tokenizer:
    def encode(self, *args, **kwargs): return [10, 11, 12, 13]


class ReplayTests(unittest.TestCase):
    def test_stable_across_hash_seeds(self):
        code = ('import sys; sys.path.insert(0,sys.argv[1]); '
                'from bidaw.trace_replay import PromptBuilder; '
                'from types import SimpleNamespace; '
                'p=PromptBuilder(SimpleNamespace(encode=lambda *a,**k:list(range(100))),"u",7); '
                'print(p.build_prompt(30,0,100)[0])')
        outputs = [subprocess.check_output([sys.executable, '-c', code, str(ROOT)],
            env=dict(os.environ, PYTHONHASHSEED=s), text=True) for s in ['1', '2']]
        self.assertEqual(*outputs)

    def test_exact_append_only_prompt_and_context_cap(self):
        b = replay.PromptBuilder(Tokenizer(), 'u', 7)
        one, n, _ = b.build_prompt(10, 0, 100)
        self.assertEqual(n, len(one))
        b.advance(10, 8)
        two, n, _ = b.build_prompt(9, 1, 100)
        self.assertEqual(len(two), 27)
        self.assertEqual(two[:len(one)], one)
        args = dict(seed=7, max_prompt_tokens=100, max_model_len=15, speed=1)
        sched, _, truncated = replay.build_schedule({'u':[dict(ts=0,qlen=10,rlen=8,rnd=0)]}, ['u'], Tokenizer(), args)
        self.assertEqual(sched['u'], [])
        self.assertEqual(len(truncated), 1)

    def run_schedule(self, schedules, parallel=1, warmup=0, fail=False):
        args = dict(parallel=parallel, warmup_seconds=warmup, salt_prefix='t',
                    no_cache=False, lmc='mock', vllm='mock', model='m')
        def sender(*a):
            if fail: raise RuntimeError('test failure')
            time.sleep(.004)
            return dict(ttft_s=.001,total_s=.004,completion_tokens=a[3])
        samples = []
        def snapshot(_):
            samples.append(time.perf_counter())
            return dict(counter=len(samples))
        return asyncio.run(replay.execute_schedule(schedules, args, sender,
            lambda *a: 0, snapshot))

    def test_global_epoch_with_more_sessions_than_workers(self):
        rows = {str(i):[dict(ts=0,qlen=1,rlen=1,rnd=0,prompt=[1],due=0)] for i in range(5)}
        records, before, after, _ = self.run_schedule(rows)
        self.assertEqual(len(records), 5)
        self.assertTrue(all(r['scheduled_s'] == 0 for r in records))
        self.assertGreater(max(r['scheduling_lag_ms'] for r in records), 5)
        self.assertEqual((before['counter'],after['counter']), (1,2))

    def test_continuous_warmup_no_replay(self):
        rows = {'u':[dict(ts=i,qlen=1,rlen=1,rnd=i,prompt=[1],due=due)
                     for i,due in enumerate([0,.04])]}
        records, before, after, _ = self.run_schedule(rows, warmup=.02)
        self.assertEqual([r['phase'] for r in records], ['warmup','measure'])
        self.assertTrue(all('error' not in r for r in records))
        self.assertEqual((before['counter'],after['counter']), (1,2))

    def test_failed_round_stops_dependent_rounds(self):
        rows = {'u':[dict(ts=i,qlen=1,rlen=1,rnd=i,prompt=[1],due=0) for i in range(2)]}
        records, *_ = self.run_schedule(rows, fail=True)
        self.assertEqual(records[1]['error'], 'previous_round_failed')


class StreamTests(unittest.TestCase):
    def response(self, completion_tokens=3):
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def raise_for_status(self): pass
            def iter_lines(self,**kwargs):
                yield 'data: {"choices":[{"text":"hello"}]}'
                yield 'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":%d}}' % completion_tokens
                yield 'data: [DONE]'
        return Response()

    def test_uses_exact_ids_ignore_eos_and_actual_usage(self):
        requests = module('requests')
        from unittest.mock import Mock
        requests.post = Mock(return_value=self.response())
        with patch.dict(sys.modules, requests=requests):
            result = replay.stream_ttft('url',[10,11],'salt',3,'model')
        self.assertEqual(result['completion_tokens'],3)
        payload = requests.post.call_args.kwargs['json']
        self.assertEqual(payload['prompt'],[10,11])
        self.assertTrue(payload['ignore_eos'])
        self.assertTrue(payload['stream_options']['include_usage'])

    def test_rejects_trace_actual_length_mismatch(self):
        requests = module('requests', post=lambda *a,**k:self.response(2))
        with patch.dict(sys.modules, requests=requests):
            with self.assertRaisesRegex(RuntimeError,'completion tokens'):
                replay.stream_ttft('url',[10,11],'salt',3,'model')


class ReportTests(unittest.TestCase):
    def test_excludes_warmup_and_uses_metric_delta(self):
        from bidaw.repro_report import summarize
        data = dict(records=[dict(phase='warmup',total_ms=999),
            dict(phase='measure',total_ms=20,offered_latency_ms=30,scheduling_lag_ms=10),
            dict(phase='measure',error='failed')], metrics_delta={
                'lmcache_mp_lookup_requested_tokens_total':100,
                'lmcache_mp_lookup_hit_l1_tokens_total':40,
                'lmcache_mp_lookup_hit_l2_tokens_total':50})
        stats=summarize(data)
        self.assertEqual(stats['n'],1)
        self.assertEqual(stats['total_ms'],20)
        self.assertEqual(stats['l1'],.4)
        self.assertEqual(stats['l2'],.5)
        self.assertEqual(stats['errors'],1)


class SchedulerTests(unittest.TestCase):
    def make_controller(self, enabled):
        from bidaw.dual_queue_scheduler import apply_dual_queue_patch
        class Controller:
            def __init__(self):
                self._submission_lock = threading.Lock()
                self._next_request_id = 1
                self._submission_queue = []
                self._pending_queue = deque()
                self._in_flight_requests = {}
                self._l2_adapters = [1]
                self.released = 0
                self.started = []
                def notify():
                    assert all(rid in self._bidaw_enqueue_monotonic for rid,_ in self._submission_queue)
                self._submission_efd = types.SimpleNamespace(notify=notify)
            def submit_prefetch_request(self,spec): return 999
            def _start_pending_requests(self): pass
            def _lock_l1_keys(self,request): pass
            def _release_all_locks(self,request): self.released += 1
            def _plan_load(self,*a): pass
            def _finish_request(self,request): pass
            def _can_admit_new_prefetch(self,spec): return True
            def _start_lookup_phase(self,rid,spec): self.started.append(rid)
        def build(rid,spec):
            return types.SimpleNamespace(key_groups=spec.key_groups,
                key_states=types.SimpleNamespace(l1_locked_keys={}), fetching_policy=spec.needs_l2)
        pc = module('prefetch_controller', PrefetchController=Controller,
                    _build_request=build, _needs_l2_lookup=lambda a,b,c:c,
                    logger=logging.getLogger('test'))
        stubs = lmcache_stubs()
        stubs['lmcache.v1.distributed.storage_controllers'] = module('storage_controllers',prefetch_controller=pc)
        with patch.dict(sys.modules, stubs), patch.dict(os.environ, BIDAW_DUAL_QUEUE=str(int(enabled))):
            apply_dual_queue_patch()
        return Controller

    def spec(self, n=1, needs_l2=True):
        return types.SimpleNamespace(key_groups=[types.SimpleNamespace(keys=[key(i) for i in range(n)])],
                                     allowed_l2_memory_tags=None, needs_l2=needs_l2)

    def test_fast_lane_has_no_timestamp_leak(self):
        C = self.make_controller(True); c = C()
        with patch('bidaw.policy_registry.feed_access') as observer:
            for _ in range(20): c.submit_prefetch_request(self.spec(needs_l2=False))
        self.assertEqual(observer.call_count, 20)
        self.assertEqual(c._bidaw_enqueue_monotonic, {})
        self.assertEqual(c._submission_queue, [])

    def test_atomic_timestamp_is_instance_local_and_hrrn_works(self):
        C = self.make_controller(True); c, other = C(), C()
        c.submit_prefetch_request(self.spec(500)); c.submit_prefetch_request(self.spec(5))
        self.assertEqual(other._bidaw_enqueue_monotonic, {})
        c._pending_queue.extend(c._submission_queue)
        c._submission_queue.clear()
        c._bidaw_enqueue_monotonic.update({1:time.monotonic()-.01,2:time.monotonic()-.008})
        c._start_pending_requests()
        self.assertEqual(c.started, [2,1])
        self.assertEqual(c._bidaw_enqueue_monotonic, {})

    def test_observation_stays_on_with_dq_disabled(self):
        C = self.make_controller(False); c = C()
        with patch('bidaw.policy_registry.feed_access') as observer:
            self.assertEqual(c.submit_prefetch_request(self.spec()),999)
        observer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
