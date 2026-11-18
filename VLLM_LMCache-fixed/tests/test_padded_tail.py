"""CPU checks for padded-tail metadata, staging and transfer completion."""
import importlib
import struct
import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from bidaw import padded_tail

TOKENS = struct.pack(">600I", *range(600))
BLOCKS = list(range(1, 39))


def module(name, **fields):
    mod = ModuleType(name)
    vars(mod).update(fields)
    return mod


class TailTests(unittest.TestCase):
    def test_600_tokens_produce_one_padded_chunk_after_512(self):
        tail = padded_tail.make_padded_tail(TOKENS, 600, BLOCKS)
        self.assertEqual((tail.start, tail.end, tail.valid_tokens), (512, 768, 88))
        self.assertEqual(tail.token_bytes[:len(TOKENS)], TOKENS)
        self.assertEqual(tail.token_bytes[len(TOKENS):], padded_tail.PADDING_TOKEN * 168)
        self.assertEqual(tail.block_ids, BLOCKS[32:38] + [38] * 10)
        self.assertTrue(padded_tail.valid_padded_key(tail, 88))
        self.assertFalse(padded_tail.valid_padded_key(tail, 87))

    def test_aligned_empty_and_missing_page_have_no_tail(self):
        self.assertIsNone(padded_tail.make_padded_tail(TOKENS, 512, BLOCKS))
        self.assertIsNone(padded_tail.make_padded_tail(TOKENS, 0, BLOCKS))
        self.assertIsNone(padded_tail.make_padded_tail(TOKENS, 600, BLOCKS[:-1]))
        self.assertIsNone(padded_tail.make_padded_tail(TOKENS, 600, [*BLOCKS[:-1], 0]))

    def test_noncomputed_tokens_are_replaced_before_storage(self):
        tail = padded_tail.make_padded_tail(TOKENS, 599, BLOCKS)
        self.assertEqual(tail.valid_tokens, 87)
        self.assertEqual(tail.token_bytes[599 * 4:600 * 4], padded_tail.PADDING_TOKEN)

    def test_server_zeroes_staging_suffix_after_copy_and_before_commit(self):
        events = []
        data = [0] * 256

        class Tensor:
            device = NS(type="cpu")
            ndim = 4
            shape = (2, 28, 256, 256)

            def __getitem__(self, dims):
                suffix = dims[2]
                return NS(zero_=lambda: data.__setitem__(suffix, [0] * (256 - suffix.start)))

        class Cache:
            device = NS(type="cuda")
            stream = NS(synchronize=lambda: events.append("sync"))
            kv_layer_groups_manager = NS(object_groups=[NS(kernel_group_indices=[0])])

            def transfer_object_group(self, block_ids, objects, **kwargs):
                events.append("copy")
                data[:] = [7] * 256

        class Transfer:
            _ctx = NS(chunk_size=256)

            def get_and_touch_context_entry(self, _instance_id):
                return NS(cache_context=Cache())

            def store_with_chunk_mask(self, key, instance_id, gpu_block_ids, event):
                context = self.get_and_touch_context_entry(instance_id).cache_context
                context.transfer_object_group(
                    gpu_block_ids, [NS(get_tensor=lambda _index: Tensor())],
                    object_group_id=0, direction="D2H",
                )
                events.append("commit")
                return b"event", True, [True]

        modules = {
            "lmcache": module("lmcache", lmcache_native=NS(TransferDirection=NS(D2H="D2H"))),
            "lmcache.logging": module("logging", init_logger=lambda _: NS(warning=lambda *args: None)),
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer": module(
                "transfer", LMCacheDrivenTransferModule=Transfer
            ),
            "lmcache.v1.platform.base.cache_context": module("context", BaseCacheContext=Cache),
        }
        with patch.dict(sys.modules, modules), patch.object(
            padded_tail, "PADDED_TAIL_STORE_ENABLED", False
        ):
            padded_tail.apply_padded_tail_store()
            tail = padded_tail.make_padded_tail(TOKENS, 600, BLOCKS)
            key = NS(**vars(tail), request_id="tail", request_configs={
                padded_tail.TAIL_VALID_TOKENS: 88,
            })
            self.assertTrue(Transfer().store_with_chunk_mask(key, 1, [tail.block_ids], b"")[1])
            self.assertEqual(data, [7] * 88 + [0] * 168)
            self.assertEqual(events, ["copy", "sync", "commit"])

            key.request_configs[padded_tail.TAIL_VALID_TOKENS] = 87
            self.assertFalse(Transfer().store_with_chunk_mask(key, 1, [tail.block_ids], b"")[1])
            self.assertEqual(events, ["copy", "sync", "commit"])
            key.request_configs.clear()
            self.assertTrue(Transfer().store_with_chunk_mask(key, 1, [tail.block_ids], b"")[1])
            self.assertEqual(data, [7] * 256)
            self.assertEqual(events[-2:], ["copy", "commit"])


class TailConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Connector:
            def request_finished(self, request, blocks):
                return True, None

            def build_connector_meta(self, output):
                self.metadata = NS(requests=[])
                self.metadata.add_request_metadata = self.metadata.requests.append
                return self.metadata

            def _get_connector_metadata(self):
                return self.metadata

            def wait_for_save(self):
                pass

            def get_finished(self, finished):
                self.reported_finished = finished
                return self.next_completion, set()

            def update_connector_output(self, output):
                pass

        request_status = NS(FINISHED_STOPPED=1, FINISHED_LENGTH_CAPPED=2, FINISHED_REPETITION=3)
        modules = {
            "lmcache.integration.vllm.lmcache_mp_connector": module(
                "connector", LMCacheMPConnector=Connector
            ),
            "lmcache.integration.vllm.lmcache_mp_metadata": module("meta", LMCacheMPRequestMetadata=NS),
            "lmcache.integration.vllm.vllm_multi_process_adapter": module("adapter", LoadStoreOp=NS),
            "lmcache.v1.multiprocess.token_codec": module(
                "codec", pack_token_ids=lambda tokens: struct.pack(f">{len(tokens)}I", *tokens)
            ),
            "vllm.distributed.kv_transfer.kv_connector.v1.base": module("base", KVConnectorRole=NS()),
            "vllm.v1.request": module("request", RequestStatus=request_status),
        }
        with patch.dict(sys.modules, modules):
            imported = importlib.import_module("bidaw.padded_tail_connector")
            cls.connector_type = imported.BidawPaddedTailConnector

    def test_finish_queues_one_store_and_holds_pages_until_both_complete(self):
        for parent_first in (True, False):
            with self.subTest(parent_first=parent_first):
                conn = self.connector_type.__new__(self.connector_type)
                conn._can_store = conn._save_decode_cache = True
                conn._tail_to_submit = []
                conn._tail_sessions = {}
                conn._worker_tails = {}
                conn._tail_completions = set()
                ended = []
                conn.scheduler_adapter = NS(end_session=ended.append)
                conn.request_trackers = {"r": NS(max_offload_tokens=None,
                                               request_configs=None, cache_salt="s")}
                request = NS(request_id="r", status=2, prompt_embeds=None,
                             num_computed_tokens=601, num_in_flight_tokens=1,
                             all_token_ids=list(range(601)))
                self.assertEqual(conn.request_finished(request, BLOCKS), (True, None))
                metadata = conn.build_connector_meta(None)
                self.assertEqual(len(metadata.requests), 1)
                tail = metadata.requests[0]
                self.assertEqual((tail.op.start, tail.op.end), (512, 768))
                self.assertEqual(tail.request_configs[padded_tail.TAIL_VALID_TOKENS], 88)
                conn.wait_for_save()
                self.assertEqual(conn.build_connector_meta(None).requests, [])
                first, second = ("r", tail.request_id) if parent_first else (tail.request_id, "r")
                conn.next_completion = {first}
                self.assertEqual(conn.get_finished({"r"}), (set(), set()))
                self.assertEqual(conn.reported_finished, {"r", tail.request_id})
                conn.next_completion = {second}
                self.assertEqual(conn.get_finished(set()), ({"r"}, set()))
                conn.update_connector_output(NS(finished_sending={"r"}))
                self.assertEqual(ended, [tail.request_id])
