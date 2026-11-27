# SPDX-License-Identifier: Apache-2.0
"""Opt-in STORE timing at the worker's native submit and acknowledgement points."""
import os
import time

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector as NativeConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from bidaw.async_jsonl import TimingLog


class LMCacheMPConnector(NativeConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        path = os.environ.get("BIDAW_VLLM_TIMING_LOG")
        self._timing_log = TimingLog(path) if path and role == KVConnectorRole.WORKER else None
        self._pending_timing = {}
        if self._timing_log is not None:
            self._native_submit_store = self.worker_adapter.submit_store_request
            self.worker_adapter.submit_store_request = self._timed_submit_store

    def _timed_submit_store(self, request_id, op, event, cache_salt="", request_configs=None):
        started = time.monotonic_ns()
        previous = self.worker_adapter.store_futures.get(request_id)
        result = self._native_submit_store(request_id, op, event, cache_salt, request_configs)
        future = self.worker_adapter.store_futures.get(request_id)
        if future is not None and future is not previous:
            round_id = (request_configs or {}).get("lmcache.bidaw_round_id")
            self._pending_timing[request_id] = (started, round_id)
        return result

    def get_finished(self, finished_req_ids):
        if getattr(self, "_timing_log", None) is None:
            return super().get_finished(finished_req_ids)
        healthy = self.worker_adapter.is_healthy
        result = super().get_finished(finished_req_ids)
        now = time.monotonic_ns()
        for request_id in self._pending_timing.keys() - self.worker_adapter.store_futures.keys():
            started, round_id = self._pending_timing.pop(request_id)
            self._timing_log.write({
                "ev": "hbm_store_done", "request_id": request_id, "round_id": round_id,
                "t_store_submit": started, "t_store_ack": now,
                "hbm_store_pin_ms": (now - started) / 1e6, "worker_healthy": healthy,
            })
        return result

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            if self._timing_log is not None:
                self._timing_log.close()
                self._timing_log = None
