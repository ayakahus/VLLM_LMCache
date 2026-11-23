# SPDX-License-Identifier: Apache-2.0
"""Opt-in group-level evidence for the pinned LMCache MP hybrid smoke test."""
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import version
import json
from pathlib import Path
import sys
from threading import Lock

SMOKE_SALT_PREFIX = "hybrid-smoke-"


def install_probe(output: Path):
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.mp_observability.event import EventType
    from lmcache.v1.mp_observability.event_bus import EventBus
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )
    from lmcache.v1.platform.base.cache_context import BaseCacheContext

    lock = Lock()
    subscribed = set()
    active = ContextVar("hybrid_smoke_retrieve", default=None)

    def record(event, **fields):
        line = json.dumps({"event": event, **fields})
        with lock, output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if event == "layout":
            print(f"[hybrid smoke] {line}", flush=True)

    def record_store(event):
        groups = defaultdict(lambda: defaultdict(set))
        for key in event.metadata["keys"]:
            identity = key.identity
            if identity.cache_salt.startswith(SMOKE_SALT_PREFIX):
                turn = key.extra_attributes.get("lmcache.bidaw_round_id")
                groups[(identity.model_name, identity.cache_salt, identity.kv_rank, turn)][
                    identity.object_group_id
                ].add(identity.chunk_hash.hex())
        for (model, salt, rank, turn), values in groups.items():
            record("store", model=model, salt=salt, rank=rank, turn=turn,
                   groups={group: sorted(hashes) for group, hashes in values.items()})

    def record_retrieve(event):
        encoded = event.metadata.get("hybrid_smoke_groups")
        if encoded is None:
            return
        groups = json.loads(encoded)
        count = sum(len(keys) for keys in groups.values())
        record("retrieve", model=event.metadata["model_name"], salt=event.metadata["cache_salt"],
               rank=event.metadata["hybrid_smoke_rank"],
               turn=event.metadata["hybrid_smoke_turn"], groups=groups,
               complete=(count > 0 and count == event.metadata["retrieved_count"]
                         and event.metadata["num_tokens"] > 0),
               request_id=event.session_id)

    native_register = LMCacheDrivenTransferModule.register_kv_cache
    native_retrieve = LMCacheDrivenTransferModule.retrieve
    native_copy = BaseCacheContext.transfer_object_group
    native_publish = EventBus.publish_on_stream

    @wraps(native_register)
    def register(self, instance_id, kv_caches, model_name, world_size, *args, **kwargs):
        native_register(self, instance_id, kv_caches, model_name, world_size, *args, **kwargs)
        bus = self._ctx.event_bus
        if not bus.enabled:
            raise RuntimeError("Hybrid smoke requires LMCache observability")
        if bus not in subscribed:
            bus.subscribe(EventType.L1_WRITE_FINISHED, record_store)
            bus.subscribe(EventType.MP_RETRIEVE_END, record_retrieve)
            subscribed.add(bus)
        registry = self._ctx.layout_desc_registry
        attn = registry.find_attn_desc(model_name, world_size)
        layouts = registry.find_group_layout_descs(model_name, world_size)
        cache = self._cache_contexts[instance_id].cache_context
        manager = cache.kv_layer_groups_manager
        groups = []
        for i, group in enumerate(manager.object_groups):
            layout = layouts[i]
            kernel_ids = group.kernel_group_indices
            groups.append({
                "group_id": i,
                "tokens_per_engine_block": [manager.kernel_groups[k].tokens_per_block for k in kernel_ids],
                "engine_blocks_per_chunk": [cache.calculate_num_blocks(self._ctx.chunk_size, k)
                                            for k in kernel_ids],
                "bytes_per_chunk": sum(shape.numel() * dtype.itemsize for shape, dtype in
                                       zip(layout.shapes, layout.dtypes, strict=True)),
            })
        record("layout", model=model_name, world_size=world_size, chunk_size=self._ctx.chunk_size,
               group_kinds=attn.group_kinds, num_chunks_in_sw=attn.num_chunks_in_sw, groups=groups)

    @wraps(native_retrieve)
    def retrieve(self, key, *args, **kwargs):
        if not key.cache_salt.startswith(SMOKE_SALT_PREFIX):
            return native_retrieve(self, key, *args, **kwargs)
        hashes = self._ctx.token_hasher.compute_packed_chunk_hashes(
            key.token_bytes, start=key.start, end=key.end
        )
        rank = ObjectKey.ComputeKVRank(
            world_size=key.world_size, global_rank=key.worker_id,
            local_world_size=key.world_size, local_rank=key.worker_id,
        )
        turn = (key.request_configs or {}).get("lmcache.bidaw_round_id", "")
        token = active.set((rank, turn, hashes, {}))
        try:
            return native_retrieve(self, key, *args, **kwargs)
        finally:
            active.reset(token)

    @wraps(native_copy)
    def copy(self, block_ids_gpu, memory_objs, **kwargs):
        native_copy(self, block_ids_gpu, memory_objs, **kwargs)
        capture = active.get()
        if capture is not None:
            _, _, hashes, groups = capture
            kept = [h.hex() for h, obj in zip(hashes, memory_objs, strict=True) if obj is not None]
            if kept:
                groups[kwargs["object_group_id"]] = kept

    @wraps(native_publish)
    def publish(self, stream, event):
        capture = active.get()
        if capture is not None and event.event_type == EventType.MP_RETRIEVE_END:
            rank, turn, _, groups = capture
            event.metadata["hybrid_smoke_rank"] = rank
            event.metadata["hybrid_smoke_turn"] = turn
            # Native CUDA stream events carry only integer and string metadata.
            event.metadata["hybrid_smoke_groups"] = json.dumps(groups)
        return native_publish(self, stream, event)

    LMCacheDrivenTransferModule.register_kv_cache = register
    LMCacheDrivenTransferModule.retrieve = retrieve
    BaseCacheContext.transfer_object_group = copy
    EventBus.publish_on_stream = publish


def main():
    if version("lmcache") != "0.5.7.post1":
        raise SystemExit("Hybrid smoke probe requires LMCache 0.5.7.post1")
    if len(sys.argv) < 3 or sys.argv[2] != "server":
        raise SystemExit("Usage: python -m bidaw.hybrid_smoke_probe EVIDENCE.jsonl server [flags]")
    output = Path(sys.argv.pop(1))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    install_probe(output)

    from lmcache.cli.main import main as lmcache_main

    lmcache_main()


if __name__ == "__main__":
    main()
