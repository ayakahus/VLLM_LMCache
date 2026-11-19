"""CPU-only regression tests. LMCache ABI is mocked, not GPU-validated."""
import asyncio
import importlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
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
from bidaw import l2_event_probe, policy_registry
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
            k, weight = rng.randrange(80), rng.randrange(256)
            prev = next((i for i in range(len(trace)-1, -1, -1) if trace[i][0] == k), None)
            expected = None if prev is None else sum(dict(trace[prev+1:]).values())
            self.assertEqual(tracker.access(k, weight), expected)
            trace.append((k, weight))

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
        a = key('A')
        self.p.on_keys_created([a])
        self.p.on_conversation_access('u', 20); self.p.on_keys_removed([a])
        self.p.on_conversation_access('v', 30); self.p.on_keys_created([a])
        self.p.on_conversation_access('u', 40)
        self.assertEqual(list(self.p._salt_hist['u']), [30])

    def test_one_sample_per_round_with_distinct_other_users(self):
        for salt, size in [('u', 0), ('v', 30), ('v', 45), ('w', 12), ('u', 20), ('u', 34)]:
            self.p.on_conversation_access(salt, size)
        self.assertEqual(self.p._seq, 6)
        self.assertEqual(list(self.p._salt_hist['u']), [57, 0])

    def test_lookup_retry_does_not_train_as_an_extra_round(self):
        self.p.on_conversation_access('u', 0, 'first')
        self.p.on_conversation_access('v', 50, 'other')
        self.p.on_conversation_access('u', 0, 'first')
        self.p.on_conversation_access('u', 32, 'second')
        self.assertEqual(self.p._logical_requests, 3)
        self.assertEqual(list(self.p._salt_hist['u']), [50])

    def test_identity_preserves_tenant_and_rank(self):
        self.assertNotEqual(self.p._key_id(key('A', 'u')), self.p._key_id(key('A', 'v')))
        self.assertNotEqual(self.p._key_id(key('A')), self.p._key_id(Key(Identity(b'A', kv_rank=1))))

    def test_ghost_builds_and_records_snapshot_sequence(self):
        self.p._ghost_window = 256
        for i in range(80):
            self.p.on_conversation_access(str(i % 20), 100)
        self.p._rebuild_ghost()
        self.assertIsNotNone(self.p._ghost)
        self.assertEqual(self.p._ghost_built_seq, 80)
        self.assertEqual(self.p._ghost[0], 1.0)

    def test_ghost_matches_slow_belady_bucket_reference(self):
        self.p._ghost_window = 256
        self.p._capacity_bytes = 40
        self.p._extreme_threshold = 240
        rng = random.Random(5)
        trace = [(str(rng.randrange(10)), rng.randrange(60)) for _ in range(180)]
        for salt, size in trace:
            self.p.on_conversation_access(salt, size)
        self.p._rebuild_ghost()
        cache, buckets = {}, [[] for _ in range(self.p._promising_buckets)]
        for i, (salt, size) in enumerate(trace):
            hit = cache.pop(salt, None) is not None and size <= 40
            prev = next((j for j in range(i-1,-1,-1) if trace[j][0] == salt),None)
            if prev is not None and size:
                distance = sum(dict(trace[prev+1:i]).values())
                kind, idx = self.p._bucket_of(distance)
                if kind == 1: buckets[idx].append(int(hit))
            if 0 < size <= 40:
                def next_use(k):
                    return next((j for j in range(i+1,len(trace)) if trace[j][0] == k),
                                float('inf'))
                while sum(cache.values()) + size > 40:
                    del cache[max(cache, key=next_use)]
                cache[salt] = size
        for i, bucket in enumerate(buckets):
            if bucket:
                self.assertAlmostEqual(self.p._ghost[1][i],sum(bucket)/len(bucket))

    def test_filter_protects_locked_keys_and_zero_ratio(self):
        a, b = key('A'), key('B')
        self.p.on_keys_created([a,b])
        self.assertEqual(self.p.get_eviction_actions(0), [])
        actions = self.p.get_eviction_actions(1, key_eligible_filter=lambda k: k != a)
        self.assertEqual([k for act in actions for k in act.keys], [b])

    def test_unsalted_eviction_respects_target_recency_and_scope(self):
        keys = [key(i, '') for i in range(10)]
        salted = key('salted', 'user')
        with patch.object(self.mod.time, 'monotonic', side_effect=[1, 2, 3, 4, 5]):
            self.p.on_keys_created(keys[:1])
            self.p.on_keys_created(keys[1:2])
            self.p.on_keys_created([*keys[2:], salted])
            self.p.on_keys_touched(keys[:1])
            actions = self.p.get_eviction_actions(
                .1, key_eligible_filter=lambda k: k != keys[1], cache_salt=''
            )
        self.assertEqual([k for action in actions for k in action.keys], [keys[2]])

    def test_salted_eviction_keeps_the_selected_user_group_together(self):
        salted = [key(i, 'user') for i in range(3)]
        unsalted = key('shared', '')
        self.p.on_keys_created([*salted, unsalted])
        actions = self.p.get_eviction_actions(.25, cache_salt='user')
        self.assertEqual([k for action in actions for k in action.keys], salted)

    def test_unsalted_access_counts_in_global_reuse_distance(self):
        unsalted = key('B', '')
        self.p.on_conversation_access('u', 100)
        self.p.on_request_access({unsalted: 20})
        self.p.on_request_access({unsalted: 20})
        self.p.on_conversation_access('u', 110)
        self.assertEqual(list(self.p._salt_hist['u']), [20])
        self.assertNotIn('', self.p._salt_hist)
        self.assertEqual(len(self.p._trace), 4)

    def test_unknown_answer_does_not_train_lower_bound(self):
        for _ in range(12): self.p._record_wrd('u', 5)
        self.assertEqual(self.p._lb_min, {})
        self.p.on_answer('u', 64)
        for _ in range(12): self.p._record_wrd('u', 5)
        self.assertEqual(self.p._lb_min, {64: 5})

    def test_unknown_answer_is_not_scored_in_the_short_answer_bin(self):
        unknown = key('a', 'unknown')
        known = key('b', 'known')
        self.p.on_keys_created([unknown, known])
        for candidate in (unknown, known):
            self.p.on_conversation_access(candidate.cache_salt, 1)
            self.p.on_conversation_access(candidate.cache_salt, 1)

        self.p.on_answer('training', 0)
        for _ in range(8):
            self.p._record_wrd('training', self.p._extreme_threshold)
        self.p.on_answer('known', 0)

        actions = self.p.get_eviction_actions(.5)
        self.assertEqual([k for action in actions for k in action.keys], [known])

    def test_lower_bound_preserves_all_observations_in_its_bucket(self):
        self.p._capacity_bytes = 16
        self.p._extreme_threshold = 96
        self.p._salt_hist['u'].extend([15, 17, 24, 27])
        self.p.on_answer('u', 64)
        self.p._lb_min[64] = 24
        self.p._ghost = (1.0, [0.5, 0.2] + [0] * 6, 0)
        self.assertAlmostEqual(self.p._potential('u', respect_lb=True), 0.4)
        self.p.on_answer('u', 65)
        self.assertAlmostEqual(self.p._potential('u', respect_lb=True), 0.55)


class Tokenizer:
    def encode(self, *args, **kwargs): return [10, 11, 12, 13]


class ReplayTests(unittest.TestCase):
    def test_stable_across_hash_seeds(self):
        code = ('import sys; sys.path.insert(0,sys.argv[1]); '
                'from bidaw.trace_replay import build_schedule; '
                'from types import SimpleNamespace; '
                'tok=SimpleNamespace(encode=lambda *a,**k:list(range(100))); '
                'args=dict(seed=7,max_prompt_tokens=100,max_model_len=110,speed=1); '
                'print(build_schedule({"u":[dict(ts=0,rnd=0,qlen=30,rlen=1)]},["u"],tok,args))')
        outputs = [subprocess.check_output([sys.executable, '-c', code, str(ROOT)],
            env=dict(os.environ, PYTHONHASHSEED=s), text=True) for s in ['1', '2']]
        self.assertEqual(*outputs)

    def test_context_cap_includes_planned_completion_tokens(self):
        users = {'u': [dict(ts=0, qlen=10, rlen=8, rnd=0),
                       dict(ts=1, qlen=9, rlen=4, rnd=1)]}
        args = dict(seed=7, max_prompt_tokens=27, max_model_len=30, speed=1)
        schedule, _, truncated = replay.build_schedule(users, ['u'], Tokenizer(), args)
        self.assertEqual(len(schedule['u']), 1)
        self.assertEqual(truncated, [dict(user='u', round=1, reason='context_limit')])
        args['max_model_len'] = 31
        schedule, _, truncated = replay.build_schedule(users, ['u'], Tokenizer(), args)
        self.assertEqual([row['prompt_tokens'] for row in schedule['u']], [10, 27])
        self.assertEqual([len(row['query_token_ids']) for row in schedule['u']], [10, 9])
        self.assertEqual(truncated, [])

    def row(self, index, due):
        return dict(ts=index, qlen=1, rlen=1, rnd=index, due=due,
                    query_token_ids=[index + 1], prompt_tokens=2 * index + 1)

    def test_shared_replay_uses_one_prefix_and_distinct_logical_sessions(self):
        users = {uid: [dict(ts=0, qlen=2, rlen=1, rnd=0),
                       dict(ts=1, qlen=2, rlen=1, rnd=1)] for uid in ("a", "b")}
        args = dict(seed=7, max_prompt_tokens=300, max_model_len=300, speed=1000,
                    shared_cache=True, shared_prefix_tokens=256, parallel=2, warmup_seconds=0,
                    salt_prefix="t", lmc="mock", vllm="mock", model="m",
                    track_chunk_fates=True, require_answer_policy=True)
        schedules, _, truncated = replay.build_schedule(users, list(users), Tokenizer(), args)
        self.assertFalse(truncated)
        requests = {}
        answers = []

        def sender(_, prompt, salt, length, model, *, session_id, round_id, history_tokens):
            requests[session_id, history_tokens] = (list(prompt), salt, round_id)
            return dict(ttft_s=.001, total_s=.002, completion_tokens=length,
                        completion_token_ids=[42] * length)

        def reporter(_, session, length):
            answers.append((session, length))
            return 1

        with patch.object(replay, "report_fate_round") as fate:
            records, *_ = asyncio.run(replay.execute_schedule(
                schedules, args, sender, reporter, lambda _: {}))

        self.assertTrue(all("error" not in record for record in records))
        self.assertEqual(requests["t-a", 0][0][:256], requests["t-b", 0][0][:256])
        self.assertTrue(all(not salt for _, salt, _ in requests.values()))
        self.assertEqual(len(requests["t-a", 259][0]), 261)
        self.assertCountEqual(answers, [("t-a", 1), ("t-b", 1)] * 2)
        ends = [call for call in fate.call_args_list if call.args[3] == "session_end"]
        self.assertCountEqual([call.args[1] for call in ends], ["t-a", "t-b"])
        self.assertTrue(all(call.kwargs["shared_cache"] for call in ends))

    def run_schedule(self, schedules, parallel=1, warmup=0, fail=False, sender=None):
        args = dict(parallel=parallel, warmup_seconds=warmup, salt_prefix='t',
                    no_cache=False, lmc='mock', vllm='mock', model='m')
        def default_sender(*a, **kwargs):
            if fail: raise RuntimeError('test failure')
            time.sleep(.004)
            return dict(ttft_s=.001, total_s=.004, completion_tokens=a[3],
                        completion_token_ids=[42] * a[3])
        samples = []
        def snapshot(_):
            samples.append(time.perf_counter())
            return dict(counter=len(samples))
        return asyncio.run(replay.execute_schedule(schedules, args, sender or default_sender,
            lambda *a: 0, snapshot))

    def test_global_epoch_with_more_sessions_than_workers(self):
        rows = {str(i): [self.row(0, 0)] for i in range(5)}
        records, before, after, _ = self.run_schedule(rows)
        self.assertEqual(len(records), 5)
        self.assertTrue(all(r['scheduled_s'] == 0 for r in records))
        self.assertGreater(max(r['scheduling_lag_ms'] for r in records), 5)
        self.assertEqual((before['counter'],after['counter']), (1,2))

    def test_continuous_warmup_no_replay(self):
        rows = {'u': [self.row(0, 0), self.row(1, .04)]}
        records, before, after, _ = self.run_schedule(rows, warmup=.02)
        self.assertEqual([r['phase'] for r in records], ['warmup','measure'])
        self.assertTrue(all('error' not in r for r in records))
        self.assertEqual((before['counter'],after['counter']), (1,2))

    def test_failed_round_stops_dependent_rounds(self):
        rows = {'u': [self.row(i, 0) for i in range(2)]}
        records, *_ = self.run_schedule(rows, fail=True)
        self.assertEqual(records[1]['error'], 'previous_round_failed')

    def test_next_prompt_includes_exact_model_tokens_across_warmup_boundary(self):
        prompts = []
        def sender(_base, prompt, _salt, max_tokens, _model, *, history_tokens):
            prompts.append((prompt, history_tokens))
            return dict(ttft_s=.001, total_s=.001, completion_tokens=max_tokens,
                        completion_token_ids=[90 + len(prompts)])
        rows = {'u': [self.row(0, 0), self.row(1, .1), self.row(2, .1)]}
        records, *_ = self.run_schedule(rows, warmup=.05, sender=sender)
        self.assertEqual(prompts, [([1], 0), ([1, 91, 2], 2), ([1, 91, 2, 92, 3], 4)])
        self.assertTrue(all('prompt_sha256' in record for record in records))

    def test_skipped_warmup_round_drops_dependent_rounds(self):
        prompts = []
        def sender(_base, prompt, _salt, max_tokens, _model, **kwargs):
            prompts.append(prompt)
            time.sleep(.15)
            return dict(ttft_s=.001, total_s=.15, completion_tokens=max_tokens,
                        completion_token_ids=[42])
        rows = {'u': [self.row(0, 0), self.row(1, .01), self.row(2, .2)]}
        records, *_ = self.run_schedule(rows, warmup=.1, sender=sender)
        self.assertEqual(prompts, [[1]])
        self.assertEqual([record.get('skip_reason') for record in records],
                         [None, 'warmup_overrun', 'missing_warmup_history'])


class StreamTests(unittest.TestCase):
    def response(self, completion_tokens=3, choices=None):
        if choices is None:
            choices = [
                dict(text='', prompt_token_ids=[10, 11], token_ids=[41]),
                dict(text='hello', token_ids=[42, 43]),
            ]
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def raise_for_status(self): pass
            def iter_lines(self,**kwargs):
                for choice in choices:
                    yield 'data: ' + json.dumps({'choices': [choice]})
                yield 'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":%d}}' % completion_tokens
                yield 'data: [DONE]'
        return Response()

    def test_uses_exact_ids_ignore_eos_and_actual_usage(self):
        requests = module('requests')
        from unittest.mock import Mock
        requests.post = Mock(return_value=self.response())
        with patch.dict(sys.modules, requests=requests), patch.object(
            replay.time, 'perf_counter', side_effect=[10, 11, 13]
        ):
            result = replay.stream_ttft('url',[10,11],'salt',3,'model',
                                        history_tokens=0, round_id='round')
        self.assertEqual(result['completion_tokens'],3)
        self.assertEqual(result['completion_token_ids'], [41, 42, 43])
        self.assertEqual(result['ttft_s'], 1)
        payload = requests.post.call_args.kwargs['json']
        self.assertEqual(payload['prompt'],[10,11])
        self.assertTrue(payload['ignore_eos'])
        self.assertTrue(payload['return_token_ids'])
        self.assertFalse(payload['add_special_tokens'])
        self.assertTrue(payload['stream_options']['include_usage'])
        self.assertEqual(payload['kv_transfer_params'], {
            'lmcache.bidaw_history_tokens': 0, 'lmcache.bidaw_round_id': 'round',
        })

    def test_rejects_trace_actual_length_mismatch(self):
        requests = module('requests', post=lambda *a,**k:self.response(2))
        with patch.dict(sys.modules, requests=requests):
            with self.assertRaisesRegex(RuntimeError,'completion tokens'):
                replay.stream_ttft('url',[10,11],'salt',3,'model')

    def test_rejects_text_only_or_incomplete_token_ids(self):
        for choice, error in [
            (dict(text='hello'), 'missing completion token IDs'),
            (dict(text='hello', token_ids=[41]), 'received 1'),
        ]:
            with self.subTest(choice=choice):
                requests = module('requests', post=lambda *a, **kw: self.response(choices=[choice]))
                with patch.dict(sys.modules, requests=requests):
                    with self.assertRaisesRegex(RuntimeError, error):
                        replay.stream_ttft('url', [10, 11], 'salt', 3, 'model')


class ReportTests(unittest.TestCase):
    def test_excludes_warmup_and_uses_metric_delta(self):
        from bidaw.repro_report import summarize
        data = dict(records=[dict(phase='warmup',total_ms=999),
            dict(phase='measure',total_ms=20,offered_latency_ms=30,scheduling_lag_ms=10),
            dict(phase='measure',skip_reason='missing_warmup_history'),
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
        self.assertEqual(stats['skipped'], 1)

    def test_comparison_rejects_different_executed_prompts(self):
        from bidaw.repro_report import main
        report = dict(label='arm', args=dict(users=1, seed=7, warmup_seconds=0, parallel=1),
                      content_mode='generated_token_prefix', schedule_sha256='same',
                      truncated_sessions=[], planned_arrival_rate=1, metrics_delta={},
                      records=[dict(phase='measure', total_ms=1,
                                    offered_latency_ms=1, scheduling_lag_ms=0)])
        with tempfile.TemporaryDirectory() as log_dir:
            for arm, digest in enumerate(('prompt-a', 'prompt-b')):
                path = Path(log_dir) / f'trace_replay_repro_v2_{arm}.json'
                path.write_text(json.dumps(dict(report, executed_prompt_sha256=digest)))
            with patch.object(sys, 'argv', ['repro_report.py', '--log-dir', log_dir]), patch('builtins.print'):
                self.assertEqual(main(), 1)
            self.assertIn('executed prompts differ', (Path(log_dir) / 'repro_v2_report.txt').read_text())


class SchedulerTests(unittest.TestCase):
    def make_controller(self, enabled):
        from bidaw.dual_queue_scheduler import apply_dual_queue_patch
        class Controller:
            def __init__(self):
                self._submission_lock = threading.Lock()
                self._stop_flag = threading.Event()
                self._next_request_id = 1
                self._submission_queue = []
                self._pending_queue = deque()
                self._in_flight_requests = {}
                self._l2_adapters = [1]
                self.released = []
                self.started = []
                self.loaded = []
                self.finished = []
                def notify():
                    assert all(rid in self._bidaw_enqueue_monotonic for rid,_ in self._submission_queue)
                self._submission_efd = types.SimpleNamespace(notify=notify)
            def submit_prefetch_request(self,spec): return 999
            def _start_pending_requests(self): pass
            def _lock_l1_keys(self,request): pass
            def _release_all_locks(self,request): self.released.append(request.request_id)
            def _plan_load(self,*a): pass
            def _finish_request(self,request): self.finished.append(request.request_id)
            def _can_admit_new_prefetch(self,spec): return True
            def _start_lookup_phase(self,rid,spec):
                self.started.append(rid)
                self._in_flight_requests[rid] = build(rid, spec)
            def _advance_request(self,request,adapters): pass
            def _poll_lookup_results(self,request,adapters): pass
            def _transition_to_load_phase(self,request):
                reserved = {0: types.SimpleNamespace(get_size=lambda: request.load_bytes)}
                self._submit_load_tasks(request, reserved if request.load_bytes else {})
            def _submit_load_tasks(self,request,reserved):
                if reserved:
                    self.loaded.append(request.request_id)
                    request.all_loads_done = lambda: False
            def _retire_request(self,request): self._in_flight_requests.pop(request.request_id)
            def _cleanup_in_flight_requests(self): pass
        def build(rid,spec):
            return types.SimpleNamespace(request_id=rid, key_groups=spec.key_groups,
                key_states=types.SimpleNamespace(l1_locked_keys={}), fetching_policy=spec.needs_l2,
                phase='lookup', load_bytes=spec.load_bytes, all_lookups_done=lambda: True,
                all_loads_done=lambda: True)
        pc = module('prefetch_controller', PrefetchController=Controller,
                    _build_request=build, _needs_l2_lookup=lambda a,b,c:c,
                    PrefetchPhase=types.SimpleNamespace(LOOKUP='lookup'),
                    get_size_bytes=lambda shapes, dtypes: shapes,
                    logger=logging.getLogger('test'))
        stubs = lmcache_stubs()
        stubs['lmcache.v1.distributed.storage_controllers'] = module('storage_controllers',prefetch_controller=pc)
        with patch.dict(sys.modules, stubs), patch.dict(os.environ, BIDAW_DUAL_QUEUE=str(int(enabled))):
            apply_dual_queue_patch()
        return Controller

    def spec(self, n=1, needs_l2=True, load_bytes=1, salt='u'):
        layout = types.SimpleNamespace(shapes=4, dtypes=None)
        group = types.SimpleNamespace(keys=[key(i, salt) for i in range(n)], layout_desc=layout)
        return types.SimpleNamespace(key_groups=[group], allowed_l2_memory_tags=None,
                                     needs_l2=needs_l2, load_bytes=load_bytes)

    def test_fast_lane_has_no_timestamp_leak(self):
        C = self.make_controller(True); c = C()
        with patch('bidaw.policy_registry.feed_access') as observer:
            for _ in range(20): c.submit_prefetch_request(self.spec(needs_l2=False))
        self.assertEqual(observer.call_count, 20)
        self.assertEqual(c._bidaw_enqueue_monotonic, {})
        self.assertEqual(c._submission_queue, [])

    def test_disk_hrrn_ranks_reserved_bytes_and_keeps_abcu_dispatch_event(self):
        C = self.make_controller(True); c, other = C(), C()
        c.submit_prefetch_request(self.spec(5, load_bytes=500))
        c.submit_prefetch_request(self.spec(500, load_bytes=5))
        c.submit_prefetch_request(self.spec(5, load_bytes=0))
        self.assertEqual(other._bidaw_enqueue_monotonic, {})
        c._pending_queue.extend(c._submission_queue)
        c._submission_queue.clear()
        self.assertEqual(c.released, [])
        c._start_pending_requests()
        self.assertEqual(c.started, [1, 2, 3])
        self.assertEqual(c.released, [1, 2, 3])
        c._bidaw_enqueue_monotonic.update({1:time.monotonic()-.01,2:time.monotonic()-.008})
        for request in list(c._in_flight_requests.values()):
            c._advance_request(request, {'lookup': {1}})
        self.assertEqual(c.loaded, [])
        self.assertEqual(c.finished, [3])
        with patch.object(l2_event_probe, '_log') as log:
            c._start_pending_requests()
        self.assertEqual(c.loaded, [2, 1])
        self.assertEqual(log.call_args_list[0].args, ('preparing', [{'request': f'{id(c)}:2'}]))
        self.assertEqual(log.call_count, 2)
        self.assertEqual(c._bidaw_disk_ready, {})
        for request in list(c._in_flight_requests.values()):
            c._retire_request(request)
        self.assertEqual(c._bidaw_enqueue_monotonic, {})

    def test_metadata_admission_skips_head_and_cleanup_releases_waiting_pins(self):
        C = self.make_controller(True); c = C()
        c.submit_prefetch_request(self.spec(20))
        c.submit_prefetch_request(self.spec(2))
        c._pending_queue.extend(c._submission_queue)
        c._can_admit_new_prefetch = lambda spec: len(spec.key_groups[0].keys) <= 10
        c._start_pending_requests()
        self.assertEqual(c.started, [2])
        self.assertEqual(c.released, [2])
        c._cleanup_in_flight_requests()
        self.assertEqual(c.released, [2, 1])
        self.assertEqual(c._bidaw_waiting_locks, {})

    def test_enqueue_error_releases_waiting_pins(self):
        C = self.make_controller(True); c = C()
        with patch.object(c._submission_efd, 'notify', side_effect=OSError('closed')):
            with self.assertRaises(OSError):
                c.submit_prefetch_request(self.spec())
        self.assertEqual(c.released, [1])
        self.assertEqual(c._bidaw_enqueue_monotonic, {})
        self.assertEqual(c._bidaw_waiting_locks, {})
        self.assertEqual(c._submission_queue, [])

    def test_observation_stays_on_with_dq_disabled(self):
        C = self.make_controller(False); c = C()
        with patch('bidaw.policy_registry.feed_access') as observer:
            self.assertEqual(c.submit_prefetch_request(self.spec(salt='')),999)
        observer.assert_called_once_with({key(0, ''): 4})


class LookupTests(unittest.TestCase):
    def test_conversation_observed_before_native_empty_chunk_exit(self):
        from bidaw.server_patch import _patch_conversation_lookup

        class Lookup:
            def lookup(self, key, tp_size):
                return 'native result'

        Lookup.lookup.handler_options = 'blocking'
        layout = types.SimpleNamespace(shapes=256, dtypes=None)
        registry = types.SimpleNamespace(find_group_layout_descs=lambda *args: {0: layout})
        lookup = Lookup()
        lookup.context = types.SimpleNamespace(chunk_size=256, layout_desc_registry=registry)
        request = types.SimpleNamespace(cache_salt='user', model_name='model',
            world_size=1, num_tokens=20, request_id='first',
            request_configs={'lmcache.bidaw_history_tokens': 0})
        stubs = {
            'lmcache.utils': module('utils', get_size_bytes=lambda shapes, dtypes: shapes),
            'lmcache.v1.multiprocess.modules.lookup': module('lookup', LookupModule=Lookup),
        }
        with patch.dict(sys.modules, stubs), patch.object(policy_registry, 'feed_conversation') as feed:
            _patch_conversation_lookup()
            self.assertEqual(Lookup.lookup.handler_options, 'blocking')
            self.assertEqual(lookup.lookup(request, 1), 'native result')
            feed.assert_called_once_with('user', 0, 'first')
            request.request_configs['lmcache.bidaw_history_tokens'] = 14
            request.request_id = 'second'
            lookup.lookup(request, 1)
            self.assertEqual(feed.call_args.args, ('user', 14, 'second'))


if __name__ == '__main__':
    unittest.main()
