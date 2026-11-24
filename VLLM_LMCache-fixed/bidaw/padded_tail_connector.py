"""Finish-time padded STORE for the pinned vLLM and LMCache MP connector."""
import json
from importlib.metadata import version
from urllib.request import urlopen
from uuid import uuid4

from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.request import RequestStatus

from bidaw.padded_tail import (
    BLOCK_TOKENS,
    CHUNK_TOKENS,
    TAIL_PARENT_REQUEST,
    TAIL_VALID_TOKENS,
    make_padded_tail,
)
from bidaw.vllm_timing_connector import LMCacheMPConnector


class BidawPaddedTailConnector(LMCacheMPConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        if version("vllm") != "0.33.0" or version("lmcache") != "0.5.7.post1":
            raise RuntimeError("Padded tails require vLLM 0.33.0 and LMCache 0.5.7.post1")
        if vllm_config.speculative_config is not None:
            raise ValueError("Padded tails require vLLM without speculative decoding")
        if role == KVConnectorRole.SCHEDULER:
            with urlopen("http://127.0.0.1:8080/bidaw/status", timeout=5) as response:
                server_status = json.load(response)
            if server_status.get("padded_tail_store") is not True:
                raise RuntimeError("Start the LMCache server with BIDAW_PAD_TAIL=1")
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.bidaw_dual_queue = server_status.get("scheduler", {}).get("enabled") is True
            if self.bidaw_dual_queue:
                # Start lookup as soon as the request enters the waiting queue.
                self._eager_prefetch = True
        adapter = self.scheduler_adapter if role == KVConnectorRole.SCHEDULER else self.worker_adapter
        if (adapter.lmcache_tokens_per_chunk != CHUNK_TOKENS
                or list(self._group_tokens_per_block) != [BLOCK_TOKENS]
                or self.lazy_offload):
            raise ValueError("Padded tails require 256-token chunks and a single 16-token KV group")

        self._tail_to_submit = []
        self._tail_sessions = {}
        self._worker_tails = {}
        self._tail_completions = set()

    def request_finished(self, request, block_ids):
        metadata = self._tail_metadata(request, block_ids)
        result = super().request_finished(request, block_ids)
        if metadata is not None:
            self._tail_to_submit.append(metadata)
            self._tail_sessions[request.request_id] = metadata.request_id
        return result

    def _tail_metadata(self, request, block_ids):
        if (not self._can_store or not self._save_decode_cache
                or request.status not in (RequestStatus.FINISHED_STOPPED,
                                          RequestStatus.FINISHED_LENGTH_CAPPED,
                                          RequestStatus.FINISHED_REPETITION)
                or request.prompt_embeds is not None):
            return None
        tracker = self.request_trackers.get(request.request_id)
        if tracker is None:
            return None
        computed = request.num_computed_tokens - request.num_in_flight_tokens
        if tracker.max_offload_tokens is not None:
            computed = min(computed, tracker.max_offload_tokens)
        tail = make_padded_tail(pack_token_ids(request.all_token_ids), computed, block_ids)
        if tail is None:
            return None
        configs = dict(tracker.request_configs or {})
        configs.update({
            TAIL_VALID_TOKENS: tail.valid_tokens,
            TAIL_PARENT_REQUEST: request.request_id,
        })
        # A separate transfer ID keeps the worker's normal STORE future intact.
        return LMCacheMPRequestMetadata(
            request_id=f"bidaw-tail-{uuid4().hex}",
            direction="STORE",
            op=LoadStoreOp(token_bytes=tail.token_bytes, block_ids=[tail.block_ids],
                           start=tail.start, end=tail.end),
            cache_salt=tracker.cache_salt,
            request_configs=configs,
        )

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        for tail in self._tail_to_submit:
            metadata.add_request_metadata(tail)
        self._tail_to_submit.clear()
        return metadata

    def wait_for_save(self):
        for metadata in self._get_connector_metadata().requests:
            parent = (metadata.request_configs or {}).get(TAIL_PARENT_REQUEST)
            if (metadata.direction == "STORE" and parent is not None
                    and metadata.request_id.startswith("bidaw-tail-")):
                self._worker_tails[parent] = metadata.request_id
        return super().wait_for_save()

    def get_finished(self, finished_req_ids):
        sending, receiving = super().get_finished(
            finished_req_ids | set(self._worker_tails.values())
        )
        sending = set(sending or ())
        for parent, tail in list(self._worker_tails.items()):
            pair = {parent, tail}
            self._tail_completions.update(sending & pair)
            sending.difference_update(pair)
            # The engine can free the parent's pages only after both stores.
            if pair <= self._tail_completions:
                self._tail_completions.difference_update(pair)
                del self._worker_tails[parent]
                sending.add(parent)
        return sending, receiving

    def update_connector_output(self, connector_output):
        super().update_connector_output(connector_output)
        for parent in connector_output.finished_sending or ():
            tail = self._tail_sessions.pop(parent, None)
            if tail is not None:
                self.scheduler_adapter.end_session(tail)
